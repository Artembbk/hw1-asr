import torch

class RemoveFreq():
    def __init__(self):
        pass

    def __call__(self, spec):
        i = torch.randint(1, spec.shape[0] - 10, (1,)).item() 
        spec[i:i+10, :] = 0
        return spec
