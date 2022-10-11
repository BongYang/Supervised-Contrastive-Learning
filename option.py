import os
import argparse

class BaseOption:
    def __init__(self, msg='argument for training'):
        self.parser = argparse.ArgumentParser(msg)
        self.parser.add_argument('--seed', type=int, default=3004, help='number of seed')
        
        self.parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
        self.parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
        self.parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
        self.parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
        self.parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
        
        # optimization
        self.parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
        self.parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

        # path
        self.parser.add_argument('--model_path', type=str, default='./save/models', help='model save path')
        self.parser.add_argument('--tb_path', type=str, default='./save/tensorboard', help='tensorboard path')
        self.parser.add_argument('--fig_path', type=str, default='./save/figure', help='figure save path')
        
        
    def parse(self):
        pass
    
class TrainOption(BaseOption):
    def __init__(self):
        super().__init__(msg='argument for model training')
        
        self.parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
        self.parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')

    def parse(self):
        opt = self.parser.parse_args()
        
        # set different learning rate from these 4 models
        if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
            opt.learning_rate = 0.01

        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

        opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                                opt.weight_decay, opt.trial)

        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)

        return opt
    
class TestOption(BaseOption):
    def __init__(self):
        super().__init__(msg='argument for model testing')
        
        self.parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
        self.parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')

    def parse(self):
        opt = self.parser.parse_args()
        
        # set different learning rate from these 4 models
        if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
            opt.learning_rate = 0.01

        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

        opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                                opt.weight_decay, opt.trial)

        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)

        return opt