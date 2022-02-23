import torch
import argparse
import Head.load_L
from resnet import load_B
import torch.utils.data.distributed
import torch.nn.functional as F
import logging
from data_load import MXFaceDataset, DataLoaderX
from util.callback import CallBackVerification, CallBackLogging
from util.logging import AverageMeter, init_logging
from config import config as cfg

def main(args):
    
    log_root = logging.getLogger()
    init_logging(log_root, cfg.rank, cfg.output)
        
    train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=cfg.local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(local_rank=cfg.local_rank, dataset=train_set, batch_size=cfg.batch_size,sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    dropout = 0.4 ## webface 
    head = Head.load_L("arc")    
    backbone = load_B(50).to(cfg.local_rank) ## R50    
    backbone.train()
    
    opt_backbone = torch.optim.SGD(params=[{'params': backbone.parameters()}],lr=cfg.lr / 512 * cfg.batch_size * world_size, momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(params=[{'params': head.parameters()}],lr=cfg.lr / 512 * cfg.batch_size * world_size, momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_header, lr_lambda=cfg.lr_func)
    
    start_epoch = 0
    total_step = int(len(train_set) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank is 0: logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(2000, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)
    
    loss = AverageMeter()
    global_step = 0
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for (img, label) in train_loader:
            global_step += 1
            features = F.normalize(backbone(img))
            loss = head.foward(features, label)
            
            opt_backbone.zero_grad()
            opt_header.zero_grad()
            loss.backward()
            opt_header.step()
            opt_backbone.step()
            
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_amp)
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step()
        scheduler_header.step()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=3, help='local_rank')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--loss', type=str, default='arcface', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args = parser.parse_args()
    main(args)
 