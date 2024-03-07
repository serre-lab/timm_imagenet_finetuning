'''
Authored by Pinyuan Feng (Tony)
Created on Mar. 07th, 2024
Last Modified on Mar. 08th, 2024
'''

import os
import argparse
import numpy as np
import pathlib
import builtins
import datetime
import torch
import torch.nn as nn
from scipy.stats import spearmanr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class str2bool(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() in ('true', 't', '1'):
            setattr(namespace, self.dest, True)
        elif values.lower() in ('false', 'f', '0'):
            setattr(namespace, self.dest, False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid value for {self.dest}: {values}")
             
def save_checkpoint_accelerator(state, is_best_acc, is_best_alignment, epoch, accelerator, args):
    '''
    /Checkpoints/
    |__resnet50
    |    |__imagenet
    |    |    |__ckpt_0.pth
    |    |    |__best.pth
    |    |__mix
    |    |__pseudo
    |...
    ''' 
     
    pathlib.Path(args.weights).mkdir(parents=True, exist_ok=True) # "/Checkpoints/"
        
    model_dir = os.path.join(args.weights, args.model_name) # "/Checkpoints/resnet50"
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        
    save_dir = os.path.join(model_dir, args.mode) # "Checkpoints/resnet50/imagenet/"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    filename = os.path.join(save_dir, "ckpt_" + str(epoch) + ".pth.tar")
    accelerator.save(state, filename)
 
    if is_best_acc:
        best_filename = os.path.join(save_dir, 'best_acc.pth.tar') # "Checkpoints/resnet50/imagenet/best_acc.pth"
        accelerator.save(state, best_filename)
        accelerator.print("The best_acc model is saved at EPOCH", str(epoch))
            
    if is_best_alignment:
        best_filename = os.path.join(save_dir, 'best_alignment.pth.tar') # "/mnt/disks/bucket/pseudo_clickme/resnet50/imagenet/best_acc.pth"
        accelerator.save(state, best_filename)
        accelerator.print("The best_alignment model is saved at EPOCH", str(epoch))
        
    rmfile = os.path.join(save_dir, "ckpt_" + str(epoch - args.ckpt_remain) + ".pth.tar")
    if accelerator.is_main_process and os.path.exists(rmfile):
        os.remove(rmfile)
        accelerator.print("Removed ", "ckpt_" + str(epoch - args.ckpt_remain) + ".pth.tar")
        
class ProgressMeterAcc(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, accelerator):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if accelerator:
            accelerator.print('  '.join(entries))
        return 

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def synchronize_between_processes(self, accelerator):
        for meter in self.meters:
            meter.synchronize_between_processes(accelerator)
        return

class AverageMeterAcc(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def synchronize_between_processes(self, accelerator):
        c = accelerator.reduce(torch.tensor(self.count, dtype=torch.float32).to(accelerator.device), reduction="mean")
        s = accelerator.reduce(torch.tensor(self.sum, dtype=torch.float32).to(accelerator.device), reduction="mean")
        self.count = int(c.item())
        self.sum = s.item()
        self.avg = self.sum / self.count
        return
        
        # t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        # dist.barrier()
        # dist.all_reduce(t)
        # t = t.tolist()
        # self.count = int(t[0])
        # self.sum = t[1]
        # # self.avg = self.sum / self.count
        # return

import math
from typing import Union, List
import torch
from torch.optim.lr_scheduler import _LRScheduler

'''
https://github.com/santurini/cosine-annealing-linear-warmup/tree/main
'''
class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            first_cycle_steps: int,
            min_lrs: List[float] = None,
            cycle_mult: float = 1.,
            warmup_steps: int = 0,
            gamma: float = 1.,
            last_epoch: int = -1,
            min_lrs_pow: int = None,
                 ):

        '''
        :param optimizer: warped optimizer
        :param first_cycle_steps: number of steps for the first scheduling cycle
        :param min_lrs: same as eta_min, min value to reach for each param_groups learning rate
        :param cycle_mult: cycle steps magnification
        :param warmup_steps: number of linear warmup steps
        :param gamma: decreasing factor of the max learning rate for each cycle
        :param last_epoch: index of the last epoch
        :param min_lrs_pow: power of 10 factor of decrease of max_lrs (ex: min_lrs_pow=2, min_lrs = max_lrs * 10 ** -2
        '''
        assert warmup_steps < first_cycle_steps, "Warmup steps should be smaller than first cycle steps"
        assert min_lrs_pow is None and min_lrs is not None or min_lrs_pow is not None and min_lrs is None, \
            "Only one of min_lrs and min_lrs_pow should be specified"
        
        # inferred from optimizer param_groups
        max_lrs = [g["lr"] for g in optimizer.state_dict()['param_groups']]

        if min_lrs_pow is not None:
            min_lrs = [i * (10 ** -min_lrs_pow) for i in max_lrs]

        if min_lrs is not None:
            assert len(min_lrs)==len(max_lrs),\
                "The length of min_lrs should be the same as max_lrs, but found {} and {}".format(
                    len(min_lrs), len(max_lrs)
                )

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lrs = max_lrs  # first max learning rate
        self.max_lrs = max_lrs  # max learning rate in the current cycle
        self.min_lrs = min_lrs  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        assert len(optimizer.param_groups) == len(self.max_lrs),\
            "Expected number of max learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))
        
        assert len(optimizer.param_groups) == len(self.min_lrs),\
            "Expected number of min learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for i, param_groups in enumerate(self.optimizer.param_groups):
            param_groups['lr'] = self.min_lrs[i]
            self.base_lrs.append(self.min_lrs[i])

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for (max_lr, base_lr) in
                    zip(self.max_lrs, self.base_lrs)]
        else:
            return [base_lr + (max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for (max_lr, base_lr) in zip(self.max_lrs, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class EarlyStopping:
    def __init__(self, threshold=10, patience=5):
        """
        Initializes the early stopping mechanism.
        :param threshold: The maximum allowed difference between train and test accuracy.
        :param patience: How many epochs to wait after the threshold is first exceeded.
        """
        self.threshold = threshold
        self.patience = patience
        self.patience_counter = 0
        self.best_diff = float('inf')

    def __call__(self, train_acc, test_acc):
        """
        Call this at the end of each epoch, providing the current train and test accuracies.
        :param train_acc: Training accuracy for the current epoch.
        :param test_acc: Testing/validation accuracy for the current epoch.
        :return: True if training should be stopped, False otherwise.
        """
        current_diff = abs(train_acc - test_acc)

        if current_diff < self.best_diff:
            self.best_diff = current_diff
            self.patience_counter = 0
        elif current_diff > self.threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True  # Stop training

        return False  # Continue training

def compute_gradient_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def compute_image_gradient_norm(images_grad, norm_type=2):
    # Compute the L2 norm for each image in the batch individually
    individual_norms = torch.norm(images_grad.view(images_grad.shape[0], -1), p=norm_type, dim=1)

    # Compute the average L2 norm for the batch
    average_norm = torch.mean(individual_norms)

    return average_norm

# # Gradient Flow
# import matplotlib.pyplot as plt
# def plot_grad_flow(named_parameters, filename):
#     ave_grads = []
#     layers = []
#     for n, p in named_parameters:
# #         p = p.detach().cpu()
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean().cpu().numpy())
# #     plt.figure(figsize=(15, 15))
#     plt.plot(ave_grads, alpha=0.3, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(xmin=0, xmax=len(ave_grads))
# #     plt.ylim(ymin=0, ymax=5)
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.savefig('../'+str(filename)+'.png')
#     plt.clf()
    
import math
class CosineAnnealingLambdaScheduler:
    def __init__(self, init_lambda, min_lambda, max_lambda, period):
        self.lambda_value = init_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.period = period

    def update_lambda(self, epoch):
        cosine_value = math.cos(math.pi * (epoch % self.period) / self.period)
        self.lambda_value = self.min_lambda + (self.max_lambda - self.min_lambda) * (1 + cosine_value) / 2

    def get_lambda(self):
        return self.lambda_value

# '''
# https://github.com/Mikoto10032/AutomaticWeightedLoss/tree/master
# '''
# class AutomaticWeightedLoss(nn.Module):
#     """automatically weighted multi-task loss

#     Params：
#         num: int，the number of loss
#         x: multi-task loss
#     Examples：
#         loss1=1
#         loss2=2
#         awl = AutomaticWeightedLoss(2)
#         loss_sum = awl(loss1, loss2)
#     """
#     def __init__(self, num=2, init_weights=[1.0, 0.5]):
#         super(AutomaticWeightedLoss, self).__init__()
#         assert num == len(init_weights), f"Please check the number of weights! num != len(init_weights)."
#         params = torch.tensor(init_weights, requires_grad=True)
#         self.params = torch.nn.Parameter(params)

#     def forward(self, *x):
#         loss_sum = 0
#         for i, loss in enumerate(x):
#             loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
#         return loss_sum

if __name__ == '__main__':
    # awl = AutomaticWeightedLoss(2)
    # print(awl.parameters())

    pass


