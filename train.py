import argparse
import os
import numpy as np
import time
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses, make_one_hot
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # 方式1 dataloader
        if args.dataset == 'apollo':
            self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        elif args.dataset == 'serm':
            self.train_loader, self.val_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        # optimizer = torch.optim.Adam(train_params, weight_decay=args.weight_decay)

        # Define Criterion
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion1 = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='ce')
        self.criterion2 = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='dice')
        self.model, self.optimizer = model, optimizer
        
        # model size
        # from torchstat import stat
        from torchsummary import summary
        # stat(model, (3,320,1280))
        summary(model.cuda(), (3,320,1280))
                
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            #多gpu
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            # patch_replication_callback(self.model)
            
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                # self.model.module.load_state_dict(checkpoint['state_dict']) 
                # for k,v in checkpoint['state_dict'].items():
                #     print(k) 
                # self.model.load_state_dict(checkpoint['state_dict'])
                # tmp = [k for k,v in checkpoint['state_dict'].items() if k!='decoder.last_conv.8.weight' and k!='decoder.last_conv.8.bias']
                model_dict = model.state_dict()
                pretrained_dict = checkpoint
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                # model.load_state_dict({k:v for k,v in checkpoint['state_dict'].items() if k!='decoder.last_conv.8.weight' and k!='decoder.last_conv.8.bias'})
                # model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        prev_time = time.time()
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            # print(image.shape)
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            if self.args.loss_type == 'diceplusce':
                loss1 = self.criterion1(output, target)
                loss2 = self.criterion2(output, make_one_hot(target.long(), num_classes=self.nclass))
                loss = loss1 + loss2
            else:
                loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)


            if self.args.loss_type == 'diceplusce':
                end_time = time.time()
                tbar.set_description('val loss: %.3f, celoss: %.4f, diceloss: %.4f, time cost: %.3f s' \
                                     % (train_loss / (i + 1), loss1.item(), loss2.item(), end_time - prev_time))
            else:
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        # train evaluate
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        IoU = self.evaluator.Mean_Intersection_over_Union()
        mIoU = np.nanmean(IoU)
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print("Acc_tr:{}, Acc_class_tr:{}, IoU_tr:{}, mIoU_tr:{}, fwIoU_tr: {}".format(Acc, Acc_class, IoU, mIoU, FWIoU))

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        prev_time = time.time()
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad(): #
                output = self.model(image)
            if self.args.loss_type == 'diceplusce':
                loss1 = self.criterion1(output, target)
                loss2 = self.criterion2(output, make_one_hot(target.long(), num_classes=self.nclass))
                loss = loss1 + loss2
            else:
                loss = self.criterion(output, target)
            val_loss += loss.item()


            if self.args.loss_type == 'diceplusce':
                tbar.set_description('val loss: %.3f, celoss: %.4f, diceloss: %.4f, time cost: %.3f s' \
                                     % (val_loss / (i + 1), loss1.item(), loss2.item(), end_time - prev_time))
            else:
                tbar.set_description('val loss: %.3f' % (val_loss / (i + 1)))

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        IoU = self.evaluator.Mean_Intersection_over_Union()
        mIoU = np.nanmean(IoU)
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % val_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='serm',
                        choices=['apollo'],
                        help='dataset name (default: apollo)')
    # parser.add_argument('--use-sbd', action='store_true', default=False,
    #                     help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=2,#                   此处修改workers
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='pretrained_models/deeplab-resnet.pth.tar')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        args.epochs = 20

    if args.batch_size is None:
        args.batch_size = 2 * len(args.gpu_ids)#                此处修改batch_size

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        args.lr = 0.01 / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    # args.resume = './run/apollo/deeplab-mobilenet/model_best.pth.tar'
    # args.ft =  True

    print(args)#打印参数信息
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()