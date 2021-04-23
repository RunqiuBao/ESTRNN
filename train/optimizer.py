from importlib import import_module

import torch.optim.lr_scheduler as lr_scheduler

def lr_func(epoch, max_lr=1, min_lr=0, warmup_epoch=100, end_epoch=500):
    if epoch<100:
        y = max_lr
    else:
        y = (5/4-epoch/400)*max_lr
    return y

class Optimizer:
    def __init__(self, para, target):
        # create optimizer
        # trainable = filter(lambda x: x.requires_grad, target.parameters())
        trainable = target.parameters()
        optimizer_name = para.optimizer
        lr = para.lr
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        # create scheduler
        milestones = para.milestones
        gamma = para.decay_gamma
        try:
            if para.lr_scheduler == "multi_step":
                self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
            elif para.lr_scheduler == "cosine":
                self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=para.end_epoch, eta_min=1e-8)
            elif para.lr_scheduler == "cosineW":
                self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2,
                                                                          eta_min=1e-8)
            elif para.lr_scheduler == "lr_bit":
                self.scheduler = lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()
