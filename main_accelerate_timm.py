'''
Title: Timm Finetuning on ImageNet
Authored by Pinyuan Feng (Tony)
Created on Mar. 07th, 2024
Last Modified on Mar. 08th, 2024
'''

import os
import gc
import random
import time
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import timm
import wandb
from accelerate import Accelerator
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader

import utils
import torchvision.datasets as datasets
from utils import accuracy, AverageMeterAcc, ProgressMeterAcc, save_checkpoint_accelerator, CosineAnnealingWithWarmup, EarlyStopping

def accelerate_logging(loggers, values, batch_size, accelerator, args, var_names=None, status="train", isLastBatch=False):
    pairs = {}
    if not var_names: var_names = [None] * len(values)
    for logger, value, var_name in zip(loggers, values, var_names):
        val = value.item()
        logger.update(val, batch_size)
        if var_name: pairs[var_name] = val

    if status == "train" or (status == "val" and isLastBatch):
        if accelerator.is_main_process and args.wandb and var_names: # just update values on the main process
            for var_name, value in zip(var_names, values):
                wandb.log(pairs)
    
def train(train_loader, model, criterion, optimizer, lr_scheduler, accelerator, args):
    train_cce_losses = AverageMeterAcc('CCE_Loss', ':.2f')
    train_top1 = AverageMeterAcc('Acc@1', ':6.3f')
    train_progress = ProgressMeterAcc(
        len(train_loader),
        [train_cce_losses, train_top1],
        prefix="Train: ")
    
    # switch to train mode
    model.train()
    
    device = accelerator.device
    
    if accelerator.is_main_process:
        pbar = tqdm(train_loader, desc="Train", position=0, leave=True)
    else:
        pbar = train_loader

    for batch_id, (images, targets) in enumerate(pbar):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # compute losses
        outputs = model(images)
        cce_loss = criterion(outputs, targets)

        # Update 
        optimizer.zero_grad()
        accelerator.backward(cce_loss)

        optimizer.step()
        lr_scheduler.step()
        
        # Log
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        var_names = ["train_cce_loss", "train_top1_acc"] if not args.evaluate else None
        loggers = [train_cce_losses, train_top1]
        values = [cce_loss, acc1[0]]
        accelerate_logging(loggers, values, images.size(0), accelerator, args, var_names=var_names)
        
        # Log learning rate
        if accelerator.is_main_process and args.wandb: # just update values on the main process
            wandb.log({"lr": lr_scheduler.get_last_lr()[0]})

        # Synchronize results
        accelerator.wait_for_everyone()
        assert accelerator.sync_gradients
        train_progress.synchronize_between_processes(accelerator) # synchronize the tensors across all tpus for every step
        
    # progress.display(len(train_loader), accelerator)
    avg_top1, avg_cce_loss = train_top1.avg, train_cce_losses.avg
    del train_cce_losses, train_top1, train_progress
    return avg_top1, avg_cce_loss

def evaluate(eval_loader, model, criterion, accelerator, args):

    eval_cce_losses = AverageMeterAcc('CCE_Loss', ':.2f')
    eval_top1 = AverageMeterAcc('Acc@1', ':6.2f')
    eval_progress = ProgressMeterAcc(
        len(eval_loader),
        [eval_cce_losses, eval_top1],
        prefix='Eval:  ')
    
    # switch to evaluate mode
    model.eval()
    
    device = accelerator.device
    status = "test" if args.evaluate else "val"
    
    if accelerator.is_main_process:
        pbar = tqdm(eval_loader, desc="Eval ", position=0, leave=True)
    else:
        pbar = eval_loader

    for batch_id, (images, targets) in enumerate(pbar):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # compute val loss
        with torch.no_grad():
            outputs = model(images)
            cce_loss = criterion(outputs, targets)

        # Log
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        var_names = ["eval_cce_loss", "eval_top1_acc"]
        loggers = [eval_cce_losses, eval_top1]
        values = [cce_loss, acc1[0]]
        accelerate_logging(loggers, values, images.size(0), accelerator, args, var_names=var_names, status=status, isLastBatch=batch_id==(len(eval_loader)-1))
        
        # Synchronize results
        accelerator.wait_for_everyone()
        eval_progress.synchronize_between_processes(accelerator) # synchronize the tensors across all tpus for every step
        
    # progress.display(len(eval_loader), accelerator)
    avg_top1, avg_cce_loss = eval_top1.avg, eval_cce_losses.avg
    del eval_cce_losses, eval_top1, eval_progress
    return avg_top1, avg_cce_loss

def test(eval_loader, model, criterion, accelerator, args):

    eval_cce_losses = AverageMeterAcc('CCE_Loss', ':.2f')
    eval_top1 = AverageMeterAcc('Acc@1', ':6.2f')
    eval_progress = ProgressMeterAcc(
        len(eval_loader),
        [eval_cce_losses, eval_top1],
        prefix='Eval_ImageNet_Acc:  ')
    
    # switch to evaluate mode
    model.eval()
    
    device = accelerator.device
    status = "test" if args.evaluate else "val"
    
    if accelerator.is_main_process:
        pbar = tqdm(eval_loader, desc="Eval ", position=0, leave=True)
    else:
        pbar = eval_loader

    for batch_id, (images, targets) in enumerate(pbar):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # compute val loss
        with torch.no_grad():
            outputs = model(images)
            cce_loss = criterion(outputs, targets)

        # Log
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        var_names = ["eval_cce_loss", "eval_top1_acc"]
        loggers = [eval_cce_losses, eval_top1]
        values = [cce_loss, acc1[0]]
        accelerate_logging(loggers, values, images.size(0), accelerator, args, var_names=var_names, status=status, isLastBatch=batch_id==(len(eval_loader)-1))
        
        # Synchronize results
        accelerator.wait_for_everyone()
        eval_progress.synchronize_between_processes(accelerator) # synchronize the tensors across all tpus for every step
        
    # progress.display(len(eval_loader), accelerator)
    avg_top1, avg_cce_loss = eval_top1.avg, eval_cce_losses.avg
    del eval_cce_losses, eval_top1, eval_progress
    return avg_top1, avg_cce_loss


def run(args):
    start_time = time.time()
    
    global best_acc
    
    best_acc = 0
    
    # Initialize accelerator
    accelerator = Accelerator(cpu=False, mixed_precision='no') # choose from 'no', 'fp8', 'fp16', 'bf16'
    
    # enable wandb
    if args.wandb and accelerator.is_main_process and not args.evaluate:
        wandb.login(key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        wandb.init(
            project="xxxxxxxxxxxxxx",  # set the wandb project where this run will be logged
            entity="serrelab",
            config={                   # track hyperparameters and run metadata
                "learning_rate": args.learning_rate,
                "architecture": args.model_name,
                "dataset": "ClickMe",
                "epochs": args.epochs,
                "mode": args.mode,
                "pretrained": args.pretrained
            }
        )
    
    # Set the random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Choose whether to use pretrained weights
    if args.pretrained:
        accelerator.print("=> using pre-trained model '{}'".format(args.model_name))
        model = timm.create_model(args.model_name, num_classes=1000, pretrained=args.pretrained)
    else:
        accelerator.print("=> creating model '{}'".format(args.model_name))
        model = timm.create_model(args.model_name, num_classes=1000, pretrained=args.pretrained)
        # model.init.xavier_normal_(w)
        # Continue training
        if args.resume or args.evaluate:
            ckpt_path = os.path.join(args.weights, args.model_name, args.mode, args.best_model + '.pth.tar')
            if os.path.isfile(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['state_dict'])
                accelerator.print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_path, checkpoint['epoch']))
            else:
                accelerator.print("=> no checkpoint found at '{}'".format(args.resume))
                return
            
     # Initialization
    if args.evaluate:
        if args.dummy:
            accelerator.print("=> Dummy data is used!")
            val_dataset = datasets.FakeData(1000, (3, 224, 224), 1000, transforms.ToTensor())
        else:
            valdir = os.path.join(args.data_dir, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,
            ]))

        test_imagenet_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory = False, drop_last = True, shuffle = False)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        model, criterion, test_imagenet_loader = accelerator.prepare(model, criterion, test_imagenet_loader)
    else:
        if args.dummy:
            accelerator.print("=> Dummy data is used!")
            train_dataset = datasets.FakeData(5000, (3, 224, 224), 1000, transforms.ToTensor())
            val_dataset = datasets.FakeData(1000, (3, 224, 224), 1000, transforms.ToTensor())
        else:
            traindir = os.path.join(args.data_dir, 'train')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
            ]))
            valdir = os.path.join(args.data_dir, 'val')
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,
            ]))

        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory = False, shuffle = False, drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers, pin_memory = False, shuffle = False, drop_last = True)

        # Instantiate optimizer and learning rate scheduler
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=False)
        steps_per_epoch = len(train_loader) 
        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer = optimizer, min_lrs = [1e-6], first_cycle_steps = args.epochs*steps_per_epoch, warmup_steps = args.warmup*steps_per_epoch, gamma = 0.9)
        
        # Accelerate wrap up
        model, optimizer, lr_scheduler, criterion, train_loader, val_loader = accelerator.prepare(
            model, optimizer, lr_scheduler, criterion, train_loader, val_loader)
    
    if args.evaluate:
        model.eval()
        test_acc, test_loss = test(test_imagenet_loader, model, criterion, accelerator, args)
        accelerator.print(f"[Test: ] Accuracy: {test_acc:.2f} %; test_cce_loss: {test_loss:.4f}")
        
        return
    else:
        if accelerator.is_main_process:
            early_stopping = EarlyStopping(threshold=5, patience=5)
            
        for epoch in range(args.epochs):
            accelerator.print('Epoch: [%d | %d]' % (epoch + 1, args.epochs))
                
            epoch_s = time.time()

            # train for one epoch
            train_acc, train_cce_loss = train(train_loader, model, criterion, optimizer, lr_scheduler, accelerator, args)
            accelerator.print(f"[Train] Acc@1: {train_acc:.4f} %, CCE Loss: {train_cce_loss:.4f}")

            # evaluate on validation set
            val_acc, val_cce_loss = evaluate(val_loader, model, criterion, accelerator, args)
            accelerator.print(f"[Val]   Acc@1: {val_acc:.4f} %, CCE Loss: {val_cce_loss:.4f}")
            
            epoch_e = time.time()
                        
            accelerator.print("Epoch {}: {} secs".format(str(epoch+1), str(int(epoch_e - epoch_s))))

            # Skip warming up stage
            if epoch <= args.warmup: continue 
            
            # save model for best_acc model
            is_best_acc = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint_accelerator({
                'epoch': epoch + 1,
                "model_name": args.model_name,
                'state_dict': unwrapped_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'mode':args.mode
            }, is_best_acc, None, epoch+1, accelerator, args)
            
            accelerator.print("")

            if accelerator.is_main_process and early_stopping(train_acc, val_acc):
                accelerator.print("Early stopping triggered")
                break
                
    if accelerator.is_main_process:
        end_time = time.time()
        accelerator.print('Total hours: ', round((end_time - start_time) / 3600, 2))
        accelerator.print("****************************** DONE! ******************************")
        
        if args.wandb:
            wandb.finish()

if __name__ == '__main__':

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    
    # create the command line parser
    parser = argparse.ArgumentParser('Harmonization PyTorch Scripts using Accelerate', add_help=False)
    parser.add_argument("-dd", "--data_dir", required = False, type = str, 
                        default = '/media/data_cifs/imagenet/',
                        help="please enter a data directory")
    parser.add_argument("-wt", "--weights", required = False, type = str,
                        default = '../checkpoints',
                        help = "please enter a directory save checkpoints")
    parser.add_argument("-bt", "--best_model", required=False, type = str,
                        default = 'best_acc', 
                        help="'best_acc'?")
    parser.add_argument("-mn", "--model_name", required = False, type = str,
                        default = 'resnet50.tv_in1k',
                        help="Please specify a model architecture according to TIMM")
    parser.add_argument("-md", "--mode", required=False, type = str,
                        default = 'imagenet', 
                        help="'pseudo', 'mix' or 'imagenet'?")
    parser.add_argument("-ep", "--epochs", required=False, type = int, default = 3,
                        help="Number of Epochs")
    parser.add_argument("-sp", "--start_epoch", required=False, type = int, default = 0,
                        help="start epoch is usually used with 'resume'")
    parser.add_argument("-bs", "--batch_size", required=False, type = int,default = 4,
                        help="Batch Size")
    parser.add_argument("-lr", "--learning_rate", required=False, type = float, default = 5e-5,
                        help="Learning Rate")
    parser.add_argument("-mt", "--momentum", required=False, type = float, default = 0.9,
                        help="SGD momentum")
    parser.add_argument("-ss", "--step_size", required=False, type = int, default = 25,
                        help="learning rate scheduler")
    parser.add_argument("-gm", "--gamma", required=False, type = float, default = 0.1,
                        help="scheduler parameters, which decides the change of learning rate ")
    parser.add_argument("-wd", "--weight_decay", required=False, type = float, default = 1e-4,
                        help="weight decay, regularization")
    parser.add_argument("-iv", "--interval", required=False, type = int, default = 2,
                        help="Step interval for printing logs")
    parser.add_argument("-nw", "--num_workers", required=False, type = int, default = 16,
                        help="number of workers in dataloader")
    parser.add_argument("-gid", "--gpu_id", required=False, type = int, default = 1,
                        help="specify gpu id for single gpu training")
    parser.add_argument("-tc", "--tpu_cores_per_node", required=False, type = int, default = 1,
                        help="specify the number of tpu cores")
    parser.add_argument("-ckpt", "--ckpt_remain", required=False, type = int, default = 5,
                        help="how many checkpoints can be saved at most?")
    parser.add_argument("-lu", "--logger_update", required=False, type = int, default = 10,
                        help="Update interval (needed for TPU training)")
    parser.add_argument("-sd", "--seed", required=False, type = int, default = 42,
                        help="Update interval (needed for TPU training)")
    parser.add_argument("-ev", "--evaluate", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to evaluate a model")
    parser.add_argument("-wu", "--warmup", required=False, type = int, default = 3,
                        help="specify warmup epochs, usually <= 5")
    parser.add_argument("-ld", "--lambda_value", required=False, type = float, default = 1.0,
                        help="specify lambda to control the weight of harmonization loss")
    parser.add_argument("-dm", "--dummy", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to use dummy data")
    parser.add_argument("-pt", "--pretrained", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to use pretrained model from TIMM")
    parser.add_argument("-rs", "--resume", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to continue (usually used with 'evaluate')")
    parser.add_argument("-gt", "--tpu", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to use Google Cloud Tensor Processing Units")
    parser.add_argument("-wb", "--wandb", required=False, type = str, default = False,
                        action = utils.str2bool,
                        help="Whether to W&B to record progress")
    
    # modify the configurations according to args parser
    args = parser.parse_args()
    # assert args.interval > 5, f"Please make sure the interval is greater than 5"
    
    run(args)