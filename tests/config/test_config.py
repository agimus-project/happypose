import torch

if torch.cuda.is_available():
    DEVICE = ['cuda']
    # Should be later changed to DEVICE = ['cpu', 'cuda']
    # See Issue #146
else:
    DEVICE = ['cpu']
