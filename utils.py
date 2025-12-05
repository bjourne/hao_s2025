from modules import *
import torch

def replace_qcfs_by_neuron(model, neuron_type):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_qcfs_by_neuron(module, neuron_type)
        if 'qcfs' in module.__class__.__name__.lower():
            if 'ParaInfNeuron_CW_ND' in neuron_type:
                if module.dim > 3:
                    model._modules[name] = ParaInfNeuron_CW_ND(
                        module.t, module.up.item()*torch.ones_like(module.rec_th_mean.cpu()),
                        module.rec_th_mean.cpu(), 0.5*module.up.item()+module.rec_in_mean.cpu()*module.t
                    )
                else:
                    model._modules[name] = ParaInfNeuron(module.t, th=module.up.item(), init_mem=0.5, dim=module.dim)
            elif 'ParaInfNeuron' in neuron_type:
                model._modules[name] = ParaInfNeuron(module.t, th=module.up.item(), init_mem=0.5, dim=module.dim)
            elif 'IFNeuron' in neuron_type:
                model._modules[name] = IFNeuron(module.t, th=module.up.item(), init_mem=0.5)

    return model


def replace_relu_by_func(model, func_type, T=8):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_relu_by_func(module, func_type, T)
        if 'relu' in module.__class__.__name__.lower() or 'qcfs' in module.__class__.__name__.lower():
            if 'RecReLU' in func_type:
                model._modules[name] = RecReLU()
            elif 'QCFS' in func_type:
                model._modules[name] = QCFS(in_ch=module.up.shape[0],up=module.up.cpu(),t=T,is_cab=True,is_relu=True)
            elif 'ParaInfNeuron_CW_ND' in func_type:
                model._modules[name] = ParaInfNeuron_CW_ND(module.t, module.up.cpu(), module.rec_th_mean.cpu(), 0.5*module.up.cpu()+module.rec_in_mean.cpu()*module.t)
            elif 'IFNeuron' in func_type:
                model._modules[name] = IFNeuron(T, th=module.up.cpu(), init_mem=0.5)

    return model


def set_calib_inf(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            set_calib_inf(module)
        if 'qcfs' in module.__class__.__name__.lower() and (module.dim > 3):
            module.cab_inf = True


def set_calib_opt(model, on=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            set_calib_opt(module, on)
        if 'qcfs' in module.__class__.__name__.lower() and (module.dim > 3):
            module.is_cab = on
