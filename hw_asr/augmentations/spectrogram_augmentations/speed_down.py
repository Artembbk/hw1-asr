import torch
from torch.nn.functional import interpolate

class SpeedDown():
    def __init__(self):
        pass

    def __call__(self, spec):
        speed_uped = interpolate(spec, scale_factor=[0.5, 1])
        return speed_uped
