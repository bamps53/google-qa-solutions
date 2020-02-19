import math
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup
from .one_cycle import OneCycleLR

class HalfCosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.T_max = T_max
        super(HalfCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % (2 * self.T_max) < self.T_max:
            cos_unit = 0.5 * (math.cos(math.pi * self.last_epoch / self.T_max) - 1)
        else:
            cos_unit = 0.5 * (math.cos(math.pi * (self.last_epoch / self.T_max - 1)) - 1)

        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * 1.0e-4
            range = math.log10(base_lr - math.log10(min_lr))
            lrs.append(10 ** (math.log10(base_lr) + range * cos_unit))
        return lrs


def get_scheduler(config, optimizer, num_epochs, num_batches):
    num_train_steps  = num_epochs * num_batches / config.train.accumulation_steps
    if config.scheduler.name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=config.scheduler.params.patience)
    elif config.scheduler.name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, config.scheduler.params.t_max, eta_min=1e-6,
                                      last_epoch=-1)
    elif config.scheduler.name == 'cosine_warmup':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.params.t_max, eta_min=1e-6,
                                      last_epoch=-1)
    elif config.scheduler.name == 'half_cosine':
        scheduler = HalfCosineAnnealingLR(optimizer, config.scheduler.params.t_max, last_epoch=-1)
    elif config.scheduler.name == 'warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.scheduler.params.warmup,
            num_training_steps=num_train_steps,
        )
    elif config.scheduler.name == 'cosine_one_cycle':
        scheduler = OneCycleLR(optimizer, n_epochs=num_epochs, n_batches=num_batches)
    else:
        scheduler = None
    return scheduler

