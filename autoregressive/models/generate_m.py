# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

from dataset.build import build_dataset

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float,
            initial_tokens: torch.Tensor, t_indices: torch.Tensor, **sampling_kwargs):
    ## cond_idx: [2*bs], initial_tokens: [2*bs, 1, dim] 
    if cfg_scale > 1.0:
        reconstructed_img, _ = model(initial_tokens, cond_idx, t_indices, input_pos)
        reconstructed_img = reconstructed_img[:,-1,:].unsqueeze(1)
        logits_combined = reconstructed_img
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        reconstructed_img, _ = model(initial_tokens, cond_idx, t_indices, input_pos)
        reconstructed_img = reconstructed_img[:,-1,:].unsqueeze(1)
        logits = reconstructed_img

    return logits # [bs, 1, cxhxw]

@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, 
             patches_number=-1, latent_shape=-1, alphas=None, betas=None, use_args=None, **sampling_kwargs):
    
    assert alphas is not None and betas is not None, "alphas and betas should not be None!"

    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond  # [2*bs]
    else:
        raise Exception("please check model type")

    max_seq_length = T_new = max_new_tokens
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.output.linear.weight.dtype)
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    datasets = build_dataset(use_args)
    real_token = datasets[0][0].to(device, non_blocking=True)
    seq = []
    inputs = []
    initial_tokens = torch.randn(max_batch_size, 1, latent_shape, dtype=torch.float, device=device)
    inputs.append(initial_tokens.clone())
    if cfg_scale > 1.0:
        initial_tokens = initial_tokens.repeat(2, 1, 1)
    seq.append(initial_tokens[:, 0, :])
    input_pos = torch.arange(0, patches_number, device=device)
    t_indices = [alphas.shape[0]-1]
    t_indices = torch.Tensor(t_indices).unsqueeze(0).repeat(initial_tokens.shape[0], 1).to(input_pos.device)
    
    pred_noise = prefill(model, cond_combined, input_pos, 
                         cfg_scale, initial_tokens, t_indices, **sampling_kwargs)
    
    if cfg_scale > 1.0:
        next_token = (torch.split(initial_tokens, len(initial_tokens) // 2, dim=0)[0] - pred_noise * betas[alphas.shape[0] - 1]) # / alphas[alphas.shape[0] - 1]
    else:
        next_token = (initial_tokens - pred_noise * betas[alphas.shape[0] - 1]) # / alphas[alphas.shape[0] - 1]
    # print(next_token.mean(),next_token.std())
    seq.append(next_token[:, 0, :])
    if cfg_scale > 1.0:
        noise = torch.split(initial_tokens, len(initial_tokens) // 2, dim=0)[0]
    else:
        noise = initial_tokens
        
    for i in range(1, alphas.shape[0]):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            t_indices = torch.tensor([j for j in range(alphas.shape[0] - 1, alphas.shape[0] - i - 2, -1)], device=device, dtype=torch.int).unsqueeze(0).repeat(initial_tokens.shape[0], 1)
            input_pos = torch.arange(0, patches_number * (i + 1), device=device)
        

            # if i == 20:
            #     next_token = real_token
            #     pred_noise = torch.randn(max_batch_size, 1, latent_shape, dtype=torch.float, device=device)
                
            noise = next_token + pred_noise
            
            input_token = alphas[alphas.shape[0] - i - 1] * next_token + betas[alphas.shape[0] - i - 1] * noise
            inputs = [alphas[j] * next_token + betas[j] * noise
                      for j in range(alphas.shape[0] - 1, alphas.shape[0] - i - 2, -1)]
            
            if cfg_scale > 1.0:
                _inputs = torch.cat(inputs,1).repeat(2, 1, 1)
            else:
                _inputs = torch.cat(inputs,1)
            # DDIM Sampling
            pred_noise = prefill(model, cond_combined, input_pos, cfg_scale, _inputs, t_indices, **sampling_kwargs)
            if cfg_scale > 1.0:
                next_token = (input_token - pred_noise * betas[alphas.shape[0] - i - 1]) # / alphas[alphas.shape[0] - i - 1]
            else:
                next_token = (input_token - pred_noise * betas[alphas.shape[0] - i - 1]) # / alphas[alphas.shape[0] - i - 1]
            print(next_token[:, 0, :].mean(),next_token[:, 0, :].std())
            seq.append(next_token[:, 0, :])
    return torch.cat([seq[1],seq[21],seq[25],seq[31]],0)
