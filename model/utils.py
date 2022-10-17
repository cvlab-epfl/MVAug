import torch

from torch.cuda.amp import custom_bwd, custom_fwd

from misc.log_utils import log

def _sigmoid(x):
  y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1-1e-4)
  return y


def shifted_sigmo(x):
  y = 1.0 / (1.0 + torch.exp(-(x*6-3)))
  
  return y

