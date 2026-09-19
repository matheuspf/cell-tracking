"""Deterministic adjoint for the registered factor-two trilinear interpolation.

CUDA's native backward atomically accumulates into BF16 buffers. This separable
adjoint gathers contributions in a fixed order, accumulating in FP32. Forward
is the original interpolate operation. Numerical parity is tested separately.
"""
import torch
from torch import nn
from torch.nn import functional as F


class _Trilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.dtype = value.dtype
        return F.interpolate(value, scale_factor=2, mode='trilinear', align_corners=False)

    @staticmethod
    def backward(ctx, gradient):
        result=gradient.float()
        for dim in (-1,-2,-3):
            size=result.shape[dim]//2
            index=torch.arange(size,device=result.device)*2
            parts=[]
            for offset, weight in [(-1,.25),(0,.75),(1,.75),(2,.25)]:
                positions=index+offset
                valid=(positions>=0)&(positions<2*size)
                weights=torch.full((size,),weight,device=result.device)
                if offset==0:weights[0]=1.
                if offset==1:weights[-1]=1.
                weights*=valid
                shape=[1]*result.ndim;shape[dim]=size
                parts.append(result.index_select(dim,positions.clamp(0,2*size-1))*weights.reshape(shape))
            result=parts[0]+parts[1]+parts[2]+parts[3]
        return result.to(ctx.dtype)


class TrilinearX2(nn.Module):
    def forward(self, value):
        return _Trilinear.apply(value)
