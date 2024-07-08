# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
# setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
# setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
import os,sys
sys.path.append(os.path.join(os.getcwd()))

import time
import argparse
import diffusers
from diffusers.models import AutoencoderKL
from autoregressive.models.gpt_adm import GPT_models
from autoregressive.models.generate_m import generate
    
def get_vae(name="sdv1-ema"):
    if name == "sdv1-ema":
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
    elif name == "sdv1-mse":
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
    elif name == "sdxl":
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sdxl-vae")
    else:
        raise NotImplementedError
    return vae

from diffusers import EulerDiscreteScheduler, DDPMScheduler, EDMDPMSolverMultistepScheduler           
    
class AddNoise:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, image, noise, timesteps):
        results = []
        for timestep in timesteps:
            result = self.alpha[timestep] * image + self.beta[timestep] * noise
            results.append(result)
        return torch.stack(results,0)
    
import numpy as np
def diffusion_noise_schedule(schedule_type="ddpm", discrete_number=32):
    assert schedule_type in ["ddpm", "edm", "rectified_flow", "r_rectified_flow"], \
        "we only support 'ddpm', 'edm', 'rectified flow' and 'rectified flow'!"
    
    span = int(1024/discrete_number)
    index = 0
    if schedule_type == "ddpm":
        diff_scheduler = DDPMScheduler(num_train_timesteps=1024)
        alpha = diff_scheduler.alphas_cumprod[::span] ** 0.5
        beta = (1 - diff_scheduler.alphas_cumprod[::span]) ** 0.5
    elif schedule_type == "edm":
        diff_scheduler = EDMDPMSolverMultistepScheduler(num_train_timesteps=1024)
        alpha = torch.Tensor([1.] * discrete_number)
        beta = diff_scheduler.sigmas[::span]
    elif schedule_type == "rectified_flow":
        alpha = torch.from_numpy(np.linspace(1.0, 1e-3, discrete_number+1))[1:]
        beta = torch.from_numpy(np.linspace(1e-3, 1.0, discrete_number+1))[1:]
    elif schedule_type == "r_rectified_flow":
        index = torch.randint(0, int(256/discrete_number), size=(1,)).item() * discrete_number
        alpha = torch.from_numpy(np.linspace(1.0, 1e-3, 256))[index:index+discrete_number]
        beta = torch.from_numpy(np.linspace(1e-3, 1.0, 256))[index:index+discrete_number]
    else:
        raise NotImplementedError
    
    return alpha, beta, [i for i in range(index,index+discrete_number,1)]

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vae_model = get_vae(args.vae)
    vae_model.to(device)
    vae_model.eval()
    print(f"VAE is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        input_size=args.input_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        discrete_number=args.discrete_number,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif "ema" in checkpoint and args.ema:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")
    alphas, betas, _ = diffusion_noise_schedule(schedule_type=args.schedule_type,
                                             discrete_number=args.discrete_number)
    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c_indices = torch.tensor(class_labels, device=device)
    latent_size = args.input_size // args.patch_size
    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, args.discrete_number * latent_size ** 2,
        alphas=alphas, betas=betas,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, patches_number = latent_size ** 2,
        latent_shape = args.in_channels * args.input_size ** 2, use_args=args
        )
    sampling_time = time.time() - t1
    
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()
    samples = vae_model.decode(index_sample.view(-1, args.in_channels, args.input_size, args.input_size) / 0.18215).sample # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample_{}.png".format(args.gpt_type), nrow=8, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.gpt_type}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--ema", action='store_true', default=False, help="whether using ema training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--input-size", type=int, default=32, help="the size of latent code")
    parser.add_argument("--patch-size", type=int, default=8, help="the size of patches")   
    parser.add_argument("--in-channels", type=int, default=4, help="the channel of latent code")
    parser.add_argument("--discrete-number", type=int, default=32, help="the number of tokens")
    parser.add_argument("--schedule-type", type=str, default="ddpm", help="the type of noise schedule")  
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--code-path", type=str, default="/data/shaoshitong/imagenet_code_c2i_flip_ten_crop_sdv1_ema/")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vae", type=str, choices=["sdv1-ema","sdv1-mse","sdxl"], default="sdv1-ema")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)