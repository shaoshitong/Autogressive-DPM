import diffusers
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, \
                        DDPMScheduler, EDMDPMSolverMultistepScheduler
import numpy as np
import torch

class AddNoise:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, image, noise, timesteps):
        results = []
        for timestep in timesteps:
            result = self.alpha[timestep] * image + self.beta[timestep] * noise
            results.append(result)
        return torch.cat(results,0)
            
def diffusion_noise_schedule(schedule_type="ddpm", discrete_number=32):
    assert schedule_type in ["ddpm", "edm", "rectified_flow"], \
        "we only support 'ddpm', 'edm' and 'rectified flow'!"
    
    span = int(1024/discrete_number)
    if schedule_type == "ddpm":
        diff_scheduler = DDPMScheduler(num_train_timesteps=1000)
        alpha = diff_scheduler.alphas_cumprod[::span] ** 0.5
        beta = (1 - diff_scheduler.alphas_cumprod[::span]) ** 0.5
    elif schedule_type == "edm":
        diff_scheduler = EDMDPMSolverMultistepScheduler(num_train_timesteps=1000)
        alpha = torch.Tensor([1.] * discrete_number)
        beta = diff_scheduler.sigmas[::span]
    elif schedule_type == "rectified_flow":
        alpha = torch.from_numpy(np.linspace(1.0, 0.0, discrete_number+1))[1:]
        beta = torch.from_numpy(np.linspace(0.0, 1.0, discrete_number+1))[1:]
    else:
        raise NotImplementedError
    
    return alpha, beta

alpha, beta = diffusion_noise_schedule(schedule_type = "rectified_flow")
print(alpha, beta)
image = torch.randn(1,3,224,224)
noise = torch.randn(1,3,224,224)
results = AddNoise(alpha,beta)(image,noise,[i for i in range(32)])
print(results.shape)