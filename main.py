# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
from random import shuffle
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from engine import train_one_epoch

# import models
import pvt
import pvt_v2
import utils
import collections

from kaggle_face_dataset2 import trainloader, valloader
from losses import ContrastiveLoss

def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--config', required=True, type=str, help='config')

    # Model parameters
    parser.add_argument('--model', default='pvt_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=255, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')



    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='../../datasets/CASIA_aligned/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='CASIA_aligned', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19','CASIA', 'CASIA_aligned'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./pvt_decoder',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--cos_margin',action='store_true')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)
    print('Using cos margin:', args.cos_margin)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    data_loader_train = trainloader
    data_loader_val = valloader



    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=2,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    #linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size()/512.0
    #args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = ContrastiveLoss(margin=1)
    output_dir = Path(args.output_dir)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'mcp.weight']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
            '''
            if 'epoch' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            '''
        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)



    if args.eval:
        #test_stats = evaluate(data_loader_val, model, device,args.cos_margin)
        #print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        model.eval()
        input = torch.rand(1, 3, 224, 224).to(device)
        print(model(input).shape)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0


    for epoch in range(args.start_epoch, args.epochs):
        model.train()  
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        print("loader len",len(data_loader_train))
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema=None, mixup_fn=None,
            set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            fp32=args.fp32_resume,
            mcp=args.cos_margin
        )
        
        #test the network after finish each epoch, to have a brief training result.
        correct_val = 0
        total_val = 0
        with torch.no_grad():#essential for testing!!!!
            for data in valloader:
                img0, img1 , labels = data
                img0, img1 , labels = img0.cuda(), img1.cuda() , labels.cuda()
                outputs = model(img0,img1,labels)

                _, predicted = torch.max(outputs.data, 1)
                #print(predicted)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        print('Accuracy of the network on the', total_val, 'val pairs in F09', ': %d %%' % (100 * correct_val / total_val))


        lr_scheduler.step(epoch)
        if args.output_dir and epoch %5 == 0:
        #if args.output_dir:
            checkpoint_paths = [output_dir / '{}_{}.pth'.format(args.model, epoch)]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        
        #lfw_accuracy, predicts = lfw_eval.eval(model)
        #lfw_masked_accuracy, predicts = lfw_masked_eval.eval(model)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     #**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters
                     }

        if args.output_dir and utils.is_main_process():
            print('writing output')
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
