import torch
import torch.nn as nn

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grl(x, lambda_=0.6):
    return GRL.apply(x, lambda_)