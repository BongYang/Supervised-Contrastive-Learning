import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn 

import tensorboard_logger as tb_logger

from networks.resnet_big import SupConResNet, LinearClassifier
from dataset import set_loader
from option import TrainOption
from utils import adjust_learning_rate, accuracy, AverageMeter

def main():
    best_acc = 0
    opt = TrainOption().parse()
    
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    # cuda device
    opt.device = torch.device(f'cuda:'+str(opt.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(opt.device)

    # dataloader
    train_loader, val_loader = set_loader(opt, is_pretrain=False)
    
    # model
    model = SupConResNet(name=opt.model)
    classifier = LinearClassifier(name=opt.model, num_classes=1)
    
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu')['model'])
    
    # criterion
    criterion = nn.CrossEntropy()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        # validate
        test_acc, test_loss = validate(val_loader, model, classifier, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
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


def train(epoch, model, classifier, train_loader, optimizer, criterion):
    model.eval()
    classifier.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    for idx, (images, labels, feat) in enumerate(train_loader):
        batchSize = labels.shape[0]
        
        images = images.cuda()
        labels = labels.cuda()
        feat = feat.cuda()
        
        with torch.no_grad():
            img_features = model.encoder(images)
        output = classifier(img_features.detach())
        loss = criterion(output, labels)
                
        acc1 = accuracy(output, labels, topk=(1,))
        losses.update(loss.item(), batchSize)
        top1.update(acc1[0], batchSize)
            
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # time meters
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print info
        if idx % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    loss=losses, top1=top1))
            sys.stdout.flush()
    
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return losses.avg, top1.avg

def validate(epoch, model, classifier, val_loader, criterion):
    model.eval()
    classifier.eval()
        
    losses = AverageMeter()
    top1 = AverageMeter()
        
    with torch.no_grad():
        for idx, (images, labels, feats) in enumerate(val_loader):
            batchSize = labels.shape[0]
        
            images = images.cuda()
            labels = labels.cuda()
            feats = feats.cuda()
            
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)
            
            acc1 = accuracy(output, labels, topk=(1,))
            losses.update(loss.item(), batchSize)
            top1.update(acc1[0], batchSize)
            
            # print info
            if idx % 100 == 0:
                print('Epoch: [{epoch}][{idx}/{length}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch=epoch, idx=idx, length=len(val_loader), loss=losses, top1=top1))
                sys.stdout.flush()
                
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    
    return losses.avg, top1.avg

if __name__ == '__main__':
    main()
    
    
