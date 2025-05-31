import argparse
from models.VGG_QCFS import vgg16_qcfs
from models.ResNet_QCFS import resnet20_qcfs, resnet34_qcfs
from models.ResNet_ReLU import resnet18, resnet34, resnet50, resnet101
import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging
import sys
from tqdm import tqdm
from utils import *
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, load_ImageNet_dataset
from torch.cuda import amp
from timm.data import Mixup
from typing import Tuple


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def init_distributed(distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    print('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode, world_size=world_size, rank=rank)
    return True, rank, world_size, local_rank
    
    
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt
    

def train_one_epoch(model, loss_fn, optimizer, train_dataloader, sim_len, local_rank, scaler=None, mixup=None, distributed=False):
    epoch_loss, lenth = 0, 0
    model.train()
    for img, label in train_dataloader:
        img = img.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)
        lenth += len(img)
        if mixup:
            img, label = mixup(img, label)
 
        img = img.unsqueeze(0).repeat(sim_len, 1, 1, 1, 1).flatten(0, 1)   

        optimizer.zero_grad()
        if scaler is not None:
            with amp.autocast():
                spikes = model(img).mean(dim=0)
                loss = loss_fn(spikes, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:     
            spikes = model(img).mean(dim=0)
            loss = loss_fn(spikes, label)
            loss.backward()
            optimizer.step()
            
        if distributed:
            vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
            epoch_loss += vis_loss.item()
        else:
            epoch_loss += loss.item()
    
    return epoch_loss/lenth


def eval_one_epoch(model, test_dataloader, sim_len, record_time=False):
    tot = torch.zeros(sim_len).cuda()
    model.eval()
    if record_time is True:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tot_time = 0
    lenth = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            spikes = 0
            img = img.to(torch.device('cuda'), non_blocking=True)
            label = label.to(torch.device('cuda'), non_blocking=True)
            lenth += len(img)
            
            img = img.unsqueeze(0).repeat(sim_len, 1, 1, 1, 1).flatten(0, 1)
            if record_time is True:
                starter.record()
                out = model(img)
                ender.record()
                torch.cuda.synchronize()
                tot_time += starter.elapsed_time(ender) / 1000
            else:
                out = model(img)  
            out = out.view(sim_len, out.shape[0]//sim_len, -1)
            
            for t in range(sim_len):
                spikes += out[t]
                tot[t] += (label==spikes.max(1)[1]).sum().item()
    
    if record_time is True:       
        return tot/lenth, tot_time/lenth
    else:
        return tot/lenth


def eval_one_epoch_IF_speed(model, test_dataloader, sim_len, trunc_lenth=1000):
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    tot_time = torch.zeros(sim_len).cuda()
    lenth = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            time_img = 0
            img = img.to(torch.device('cuda'), non_blocking=True)
            label = label.to(torch.device('cuda'), non_blocking=True)
            lenth += len(img)
            
            for t in range(sim_len):                                 
                starter.record()
                out = model(img)
                ender.record()
                torch.cuda.synchronize()
                time_img += starter.elapsed_time(ender) / 1000
                tot_time[t] += time_img  
                
            if lenth >= trunc_lenth:
                break
          
        return torch.stack([tot_time[i] for i in range(sim_len) if i > 0 and (i & (i + 1)) == 0], dim=0) / trunc_lenth

    
def calib_one_epoch(model, dataloader):
    set_calib_opt(model, True)
    model.eval()
    acc, lenth = 0, 0
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img = img.to(torch.device('cuda'), non_blocking=True)
            label = label.to(torch.device('cuda'), non_blocking=True)
            lenth += len(img)
            img = img.unsqueeze(0).repeat(2, 1, 1, 1, 1).flatten(0, 1) 
            out = model(img)
            out = out.view(2, out.shape[0]//2, -1)[0]
            acc += (label==out.max(1)[1]).sum().item()
            
    set_calib_opt(model, False)
    set_calib_inf(model)
    return acc/lenth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='/home/cifar100/', help='Directory where the dataset is saved')
    parser.add_argument('--savedir', type=str, default='/home/', help='Directory where the model is saved')
    parser.add_argument('--trainsnn_epochs', type=int, default=300, help='Training Epochs of SNNs')
    parser.add_argument('--neuron_type', type=str, default='ParaInfNeuron')
    parser.add_argument('--net_arch', type=str, default='', help='Network Architecture')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--time_step', type=int, default=4, help='Training Time-steps for SNNs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--calibrate_th', action='store_true', default=False)
    parser.add_argument('--direct_inference', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--dev', type=str, default='0')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--distributed_init_mode', type=str, default='env://')
    parser.add_argument("--sync_bn", action="store_true", help="Use sync batch norm")
    parser.add_argument('--checkpoint_path', type=str, default="")
    parser.add_argument('--pretrained_model', action='store_true', help='Use Pretrained Models')
    parser.add_argument('--mixup', action='store_true', help='Mixup')
    parser.add_argument('--amp', action='store_true', help='Use AMP training')
    parser.add_argument('--warm-up', type=str, nargs='+', default=[], help='--warm-up <epochs> <start-factor>')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    
    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    
    log_dir = args.savedir + args.dataset + '-' + args.net_arch + '-T' + str(args.time_step)
    identifier = args.neuron_type + '_lr' + str(args.lr) + '_wd' + str(args.wd) + '_epoch' + str(args.trainsnn_epochs) + '_mixup_' + str(args.mixup)
    save_name_suffix = log_dir + '/' + identifier
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    
    distributed, rank, world_size, local_rank = init_distributed(args.distributed_init_mode)

    if args.dataset == 'CIFAR10':
        train_dataloader, test_dataloader, train_sampler, test_sampler, snn_test_dataloader = PreProcess_Cifar10(args.datadir, args.batchsize, distributed, is_cab=True)
        cls = 10
    elif args.dataset == 'CIFAR100':
        train_dataloader, test_dataloader, train_sampler, test_sampler, snn_test_dataloader = PreProcess_Cifar100(args.datadir, args.batchsize, distributed, is_cab=True)
        cls = 100
    elif args.dataset == 'ImageNet':
        train_dataloader, test_dataloader, train_sampler, test_sampler, snn_test_dataloader = load_ImageNet_dataset(args.batchsize, os.path.join(args.datadir, 'train'), os.path.join(args.datadir, 'val'), distributed, is_cab=True)
        cls = 1000
    elif local_rank == 0:
        print('unable to find dataset ' + args.dataset)
    
    is_relu = False
    if args.net_arch == 'vgg16_qcfs':
        model = vgg16_qcfs(args.time_step, args.calibrate_th, cls)
    elif args.net_arch == 'resnet20_qcfs':
        model = resnet20_qcfs(args.time_step, args.calibrate_th, cls)
    elif args.net_arch == 'resnet34_qcfs':
        model = resnet34_qcfs(args.time_step, args.calibrate_th, cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(pretrained=False if len(args.checkpoint_path) > 0 else True)
        is_relu = True
    elif args.net_arch == 'resnet34':
        model = resnet34(pretrained=False if len(args.checkpoint_path) > 0 else True)
        is_relu = True 
    elif args.net_arch == 'resnet50':
        model = resnet50(pretrained=False if len(args.checkpoint_path) > 0 else True)
        is_relu = True
    elif args.net_arch == 'resnet101':
        model = resnet101(pretrained=False if len(args.checkpoint_path) > 0 else True)
        is_relu = True
    elif local_rank == 0:
        print('unable to find model ' + args.net_arch)

    if local_rank == 0:
        #print(model)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'total parameters: {total_params} M.')
    
    if not distributed:
        if len(args.checkpoint_path) > 0:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            if args.pretrained_model is True:
                model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint['model'], strict=False)

        if is_relu is True:
            model.cuda()
            acc = eval_one_epoch(model, test_dataloader, 1)
            model = replace_relu_by_func(model, 'RecReLU')
            eval_one_epoch(model, train_dataloader, 1)

            if args.neuron_type == "ParaInfNeuron_CW_ND":
                model = replace_relu_by_func(model, 'QCFS', args.time_step)
                model.cuda()
                calib_one_epoch(model, train_dataloader)
                acc2 = eval_one_epoch(model, test_dataloader, 1)
                model = replace_relu_by_func(model, args.neuron_type)
                #print(model)
                model.cuda()
                acc3, t3 = eval_one_epoch(model, snn_test_dataloader, args.time_step, True)
                logger.info(f"SNNs Inference: Test Acc: {acc} | {acc2} | {acc3}, Speed: {t3} (T={args.time_step})")
            else:
                model = replace_relu_by_func(model, args.neuron_type)
                #print(model)
                model.cuda()
                t2 = eval_one_epoch_IF_speed(model, snn_test_dataloader, args.time_step)
                logger.info(f"SNNs Inference: Test Acc: {acc}, Speed: {t2} (T={args.time_step})")
                
            sys.exit()
            
        if args.calibrate_th is True and is_relu is False:                
            model.cuda()
            calib_one_epoch(model, train_dataloader)
            new_acc = eval_one_epoch(model, test_dataloader, 1)
            logger.info(f"Calibrate Inference: Test Acc: {new_acc}")
                       
        if args.direct_inference is True and is_relu is False:
            model.cuda()
            acc = eval_one_epoch(model, test_dataloader, 1)
            model = replace_qcfs_by_neuron(model, args.neuron_type)
            print(model)    
            model.cuda()
            if "ParaInfNeuron" in args.neuron_type:
                new_acc, t1 = eval_one_epoch(model, snn_test_dataloader, args.time_step, True)
                logger.info(f"SNNs Inference: Test Acc: {acc} | {new_acc}, Speed: {t1} (T={args.time_step})")
            else:
                t1 = eval_one_epoch_IF_speed(model, snn_test_dataloader, args.time_step)
                logger.info(f"SNNs Inference: Test Acc: {acc}, Speed: {t1} (T={args.time_step})")
            
            sys.exit()
    
    model.cuda()
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    mixup = None
    if args.mixup:
        mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=cls)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.trainsnn_epochs)

    if len(args.warm_up) != 0:
        assert len(args.warm_up) == 2
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                              start_factor=float(args.warm_up[1]),
                                              total_iters=int(args.warm_up[0])),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.trainsnn_epochs-int(args.warm_up[0])), ])

    model_without_ddp = model
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_acc1 = checkpoint['max_acc1']
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(max_acc1, start_epoch)
    else:
        start_epoch = 0
        max_acc1 = 0

    for epoch in range(start_epoch, args.trainsnn_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        epoch_loss = train_one_epoch(model, loss_fn, optimizer, train_dataloader, 1, local_rank, scaler, mixup, distributed)
        scheduler.step()

        if local_rank == 0:
            acc = eval_one_epoch(model, test_dataloader, 1)
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'max_acc1': acc[-1].item()
                }
            if max_acc1 < acc[-1].item():
                max_acc1 = acc[-1].item()
                torch.save(checkpoint, save_name_suffix + '_best_checkpoint.pth')
            torch.save(checkpoint, save_name_suffix + '_current_checkpoint.pth')

            logger.info(f"SNNs training Epoch {epoch}: Val_loss: {epoch_loss}")
            logger.info(f"SNNs training Epoch {epoch}: Test Acc: {acc} Best Acc: {max_acc1}")
        
        if distributed:
            torch.distributed.barrier()
