import torch
from torch import nn
import math

class ParaInfNeuron(nn.Module):
    def __init__(self, T, th=1., init_mem=0.5, dim=5):
        super(ParaInfNeuron, self).__init__()
        self.T = T
        self.v_threshold = th
        self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1))
        self.register_buffer('bias', (init_mem * th) / torch.arange(1, T+1).unsqueeze(-1))
        self.dim = dim
        
    def forward(self, x):
        # x.shape: [TxB, C, H, W] or [TxB, N]

        x = x.view((self.T, int(x.shape[0] / self.T)) + x.shape[1:])       
        if self.dim == 5:
            T, B, C, H, W = x.size()
            return (((self.TxT * x.view(T, -1).mean(dim=0) + self.bias) >= self.v_threshold).float() * self.v_threshold).view(-1, C, H, W)
        else:
            T, B, N = x.size()
            return (((self.TxT * x.view(T, -1).mean(dim=0) + self.bias) >= self.v_threshold).float() * self.v_threshold).view(-1, N)
        
        
class IFNeuron(nn.Module):
    def __init__(self, T, th=1., init_mem=0.5):
        super(IFNeuron, self).__init__()
        self.T = T
        self.t = 0
        if isinstance(th, torch.Tensor):
            self.register_buffer('v_threshold', th)
            self.register_buffer('v', init_mem * th)
        else:
            self.v_threshold = th
            self.v = init_mem * th
        self.init_mem = init_mem

    def forward(self, x):
        self.t += 1
        self.v = self.v + x
        spike = (self.v >= self.v_threshold).float() * self.v_threshold
        self.v = self.v - spike
        if self.t == self.T:
            self.reset()
        return spike

    def reset(self):
        self.v = self.init_mem * self.v_threshold
        self.t = 0
    

class ParaInfNeuron_CW_ND(nn.Module):
    def __init__(self, T, pre_th, post_bias, bias):
        super(ParaInfNeuron_CW_ND, self).__init__()
        self.T = T
        self.register_buffer('pre_th', pre_th) # shape: [C, 1, 1]
        self.register_buffer('post_th', pre_th + post_bias) # shape: [C, 1, 1]
        self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # shape: [T, 1, 1, 1, 1]
        self.register_buffer('bias', bias / torch.arange(1, T+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # shape: [T, 1, C, 1, 1]
        
    def forward(self, x):
        # x.shape: [TxB, C, H, W]
        
        x = x.view((self.T, int(x.shape[0] / self.T)) + x.shape[1:])
        T, B, C, H, W = x.size()
        return (((self.TxT * x.mean(dim=0) + self.bias) >= self.pre_th).float() * self.post_th).view(-1, C, H, W)


class FloorLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        
qcfs = FloorLayer.apply


class QCFS(nn.Module):
    def __init__(self, in_ch=1, up=8., t=4, dim=5, is_cab=False, is_relu=False):
        super().__init__()
        if isinstance(up, torch.Tensor):
            self.register_buffer('up', up)
        else:
            self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        self.in_ch = in_ch
        self.dim = dim
        self.is_cab = is_cab and (dim>3)
        self.is_relu = is_relu
        self.cab_inf = False
        if self.is_cab is True:
            self.register_buffer('rec_in_mean', torch.zeros(self.in_ch, 1, 1))
            self.register_buffer('rec_th_mean', torch.zeros(self.in_ch, 1, 1))
            self.is_cab = False

    def forward(self, x):
        if self.cab_inf is True:
            return torch.clamp(qcfs((x+self.rec_in_mean)*self.t/self.up+0.5)/self.t,0,1)*(self.up+self.rec_th_mean)
        elif self.is_cab is True:
            x = x.view((2, x.shape[0]//2) + x.shape[1:])
            err = (x[1] - x[0]).transpose(0, 1).flatten(1)
            self.rec_in_mean = 0.99 * self.rec_in_mean + 0.01 * err.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

            x[0] = torch.clamp(qcfs((x[0]+self.rec_in_mean)*self.t/self.up+0.5)/self.t,0,1)*(self.up+self.rec_th_mean)
            x[1] = torch.clamp(qcfs(x[1]*8/self.up+0.5)/8,0,1)*self.up if self.is_relu is False else torch.clamp(x[1], torch.zeros_like(self.up), self.up) # resnet_qcfs 8, vgg16_qcfs 4 or 16.
            
            err = (x[1] - x[0]).transpose(0, 1).flatten(1)
            self.rec_th_mean = 0.99 * self.rec_th_mean + 0.01 * err.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
            
            return x.flatten(0, 1)
        else:
            x = x / self.up
            x = qcfs(x*self.t+0.5) / self.t
            x = torch.clamp(x, 0, 1)
            return x * self.up


class RecReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_up = False
            
    def forward(self, x):
        max_th = x.transpose(0, 1).flatten(1).max(1)[0].unsqueeze(-1).unsqueeze(-1)
        if self.init_up is False:
            self.init_up = True
            self.register_buffer('up', max_th)
        else:
            self.up = torch.max(self.up, max_th)
        
        return torch.clamp(x, torch.zeros_like(self.up), self.up)
       