import torch

# merge torch.nested or torch.masked API in the long run.
# Maybe we can let inherit torch.Tensor, but seems very complex https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
class MaskedTensor(torch.masked.MaskedTensor):
    
    def __init__(self, data, mask, requires_grad=True): # ?? TODO requires_grad
        super().__init__(data, mask, requires_grad) # True for not masked, False for masked
        self.clearmaskedvalue()

    def clearmaskedvalue(self, val=0):
        self.data.masked_fill_(torch.logical_not(self.get_mask()), val)