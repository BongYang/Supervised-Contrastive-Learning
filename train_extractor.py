import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger

from utils import adjust_learning_rate, AverageMeter
from dataset import set_loader
from option import TrainOption
from networks.resnet_big import SupConResNet
from losses import SupConLoss

def main():
    best_acc = 0
    opt = TrainOption().parse()
    
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    # cuda device
    opt.device = torch.device(f'cuda:'+str(opt.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(opt.device)
    
    # dataloader
    train_loader, _ = set_loader(opt, is_pretrain=True)
    
    # model
    model = SupConResNet()
    
    # criterion
    criterion = SupConLoss(temperature=opt.temp)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_loss', train_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'loss': train_loss,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # best accuracy
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


def train(epoch, model, train_loader, optimizer, criterion):
    model.train()
    criterion.train()
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    for idx, (images, labels, feats) in enumerate(train_loader):
        batchSize = labels.shape[0]
        
        images = images.cuda()
        labels = labels.cuda()
        feats = feats.cuda()
        
        images = torch.cat([images[0], images[1]], dim=0)
        
        imgs_features = model(input)
        f1, f2 = torch.split(imgs_features, [batchSize, batchSize], dim=0)
        imgs_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(imgs_features, labels)
                
        losses.update(loss.item(), batchSize)
            
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # time meters
        end = time.time()
        batch_time.update(time.time() - end)
        
        # print info
        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, idx, len(train_loader), 
                    batch_time=batch_time, loss=losses))
            sys.stdout.flush()
    
    return losses.avg

if __name__ == '__main__':
    main()
    
    
