import diffusers
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, \
                        DDPMScheduler, EDMDPMSolverMultistepScheduler
import numpy as np
import torch

p = torch.arange(0,10)

q = p.unsqueeze(1).repeat(1,5)

print(q)

print(q.view(-1))