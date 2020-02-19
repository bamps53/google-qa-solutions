import os
import copy
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from apex import amp
from tqdm import tqdm
import wandb

from utils import to_device, to_cpu

def update_avg(curr_avg, val, idx):
    return (curr_avg * idx + val) / (idx + 1)

class Learner():
    def __init__(self, config, model, criterion, optimizer, scheduler, loaders, metric_fn, fold, train_only_head):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.metric_name = config.metric.name
        self.metric_fn = metric_fn
        self.monitor_metric = config.metric.monitor
        self.minimize_score = config.metric.minimize
        self.fold = fold
        self.device = config.device
        self.fp16 = config.train.fp16
        self.best_epoch = -1
        self.best_score = 1e6 if self.minimize_score else -1e6
        self.num_epochs = config.train.warmup_num_epochs if train_only_head else config.train.num_epochs

    @property
    def best_checkpoint_file(self): 
        return f'{self.config.work_dir}/best_model_fold{self.fold}.pth'

    def train(self):
        self.model.to(self.device)
        if self.config.train.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model,
                self.optimizer,
                opt_level="O1",
                verbosity=0
                )

        for epoch in range(self.num_epochs):
            # print('epoch {}: \t Start training...'.format(epoch))
            self.train_preds, self.train_targets = [], []
            self.model.train()
            train_loss, train_metrics = self.train_epoch()
            print(self._get_metric_string(epoch, train_loss, train_metrics))

            self.validate(epoch)

        self._on_training_end()

    def validate(self, epoch):
        # print('epoch {}: \t Start validation...'.format(epoch))

        self.valid_preds, self.valid_targets = [], []
        self.model.eval()
        val_score, val_loss, val_metrics = self.valid_epoch()
        print(self._get_metric_string(epoch, val_loss, val_metrics, 'valid'))

        if ((self.minimize_score and (val_score < self.best_score)) or 
            ((not self.minimize_score) and (val_score > self.best_score))):
            self.best_score, self.best_epoch = val_score, epoch
            self.save_model(self.best_checkpoint_file)
            print('best model: epoch {} - {:.5}'.format(epoch, val_score))
        else:
            print(f'model not improved for {epoch-self.best_epoch} epochs')

    def train_epoch(self):
        tqdm_loader = tqdm(self.loaders['train'])
        curr_loss_avg = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = self.to_device(inputs), targets.to(self.device)
            preds, loss = self.train_batch(inputs, targets, batch_idx)

            self.train_preds.append(to_cpu(preds))
            self.train_targets.append(to_cpu(targets))

            curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)

            base_lr = self.optimizer.param_groups[0]['lr']
            tqdm_loader.set_description('loss: {:.4} base_lr: {:.6}'.format(
                round(curr_loss_avg, 4), round(base_lr, 6)))

        metric_score = self.metric_fn(
            torch.cat(self.train_preds), torch.cat(self.train_targets)).item()

        return curr_loss_avg, {self.metric_name: metric_score}
    
    def valid_epoch(self):
        tqdm_loader = tqdm(self.loaders['valid'])
        curr_loss_avg = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
                with torch.no_grad():
                    inputs, targets = self.to_device(inputs), targets.to(self.device)
                    preds, loss = self.valid_batch(inputs, targets)

                    self.valid_preds.append(to_cpu(preds))
                    self.valid_targets.append(to_cpu(targets))

                    curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)
                    
                    tqdm_loader.set_description('loss: {:.4}'.format(round(curr_loss_avg, 4)))

        metric_score = self.metric_fn(
            torch.cat(self.valid_preds), torch.cat(self.valid_targets)).item()
        if self.monitor_metric: score = metric_score
        else: score = curr_loss_avg

        return score, curr_loss_avg, {self.metric_name: metric_score}
    
    def train_batch(self, batch_inputs, batch_targets, batch_idx):
        preds, loss = self.get_loss_batch(batch_inputs, batch_targets)

        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (batch_idx % self.config.train.accumulation_steps) == 0:

            clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return preds, loss.item()

    def valid_batch(self, batch_inputs, batch_targets):
        preds, loss = self.get_loss_batch(batch_inputs, batch_targets)
        return preds, loss.item()
    
    def get_loss_batch(self, batch_inputs, batch_targets):
        preds = self.model(**batch_inputs)
        loss = self.criterion(preds, batch_targets)
        return preds, loss

    def to_device(self, xs):
        return to_device(xs, self.device)
    
    def load_best_model(self):
        checkpoint = torch.load(self.best_checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_model(self, checkpoint_file):
        torch.save({'model_state_dict': self.model.state_dict()}, checkpoint_file)

    def _get_metric_string(self, epoch, loss, metrics, stage='train'):
        base_str = 'epoch {}/{} \t {} : loss {:.5}'.format(
            epoch, self.num_epochs, stage, loss)
        metrics_str = ''.join(' - {} {:.5}'.format(k, v) for k, v in metrics.items())
        wandb.log({
        '{}_loss'.format(stage):loss,
        '{}_score'.format(stage):list(metrics.values())[0],
        })
        return base_str + metrics_str

    def _on_training_end(self):
        print('TRAINING END: Best score achieved on epoch '
                  f'{self.best_epoch} - {self.best_score:.5f}')
