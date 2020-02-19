import numpy as np
import torch
import gc
from scipy.stats import spearmanr
from tqdm import tqdm
import wandb
from apex import amp

class Runner():
    def __init__(self, config, model, criterion, optimizer, scheduler, loaders, fold):
        self.config=config
        self.model=model
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.loaders=loaders
        self.fold=fold

    def train(self, train_only_head=False):
        self.model.train()
        iteration = 0
        best_score = -1.0

        if train_only_head:
            num_epochs = self.config.train.warmup_num_epochs
            for n, p in list(self.model.named_parameters()):
                if "bert" in n:
                    p.requires_grad = False
        else:
            num_epochs = self.config.train.num_epochs
            # for n, p in list(self.model.named_parameters()):
            #     p.requires_grad = True

        if config.train.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

        for epoch in range(num_epochs):
            avg_loss = 0.0
            self.optimizer.zero_grad()
            for idx, batch in enumerate(tqdm(self.loaders['train'], desc="Train", ncols=80)):
                logits, labels = process_batch(self.config.model.num_bert, self.model, batch, self.criterion)
                loss = self.criterion(logits, labels)

                if self.config.train.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if (iteration + 1) % self.config.train.accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                iteration += 1

                avg_loss += loss.item() / (len(self.loaders['train']) * self.config.train.accumulation_steps)

            avg_val_loss, val_score, val_preds = self.evaluate()

            print(
                "Fold {}/{}: \t Epoch {}/{}: \t loss={:.4f} \t val_loss={:.4f} \t val_score={:.6f}".format(
                    self.fold,
                    self.config.data.num_folds,
                    epoch + 1,
                    num_epochs,
                    avg_loss,
                    avg_val_loss,
                    val_score
                )
            )
            wandb.log({
                'train_loss':avg_loss,
                'valid_loss':avg_val_loss,
                'valid_score':val_score,
                })

            if val_score > best_score:
                best_score = val_score
                torch.save(
                    self.model.state_dict(),
                    self.config.work_dir + '/best_model_fold{}.pth'.format(self.fold),
                )
                np.save(self.config.work_dir + '/val_preds_fold{}'.format(self.fold), val_preds)

    def evaluate(self):
        avg_val_loss = 0.0
        self.model.eval()

        valid_preds = []
        original = []
        ids = []

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.loaders['valid'], desc="Valid", ncols=80)):
                logits, labels = process_batch(self.config.model.num_bert, self.model, batch, self.criterion)
                loss = self.criterion(logits, labels)

                avg_val_loss += loss.item() / len(self.loaders['valid'])
                valid_preds.extend(logits.detach().cpu().numpy())
                original.extend(labels.detach().cpu().numpy())

            valid_preds = np.array(valid_preds)
            original = np.array(original)

            score = 0
            preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()

            for i in range(len(self.config.data.output_columns)):
                score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)

        return avg_val_loss, score / len(self.config.data.output_columns), preds[np.argsort(ids)]


def process_batch(num_bert, model, batch, criterion, return_logits=False):
    if num_bert == 1:
        input_ids = batch["input_ids"].cuda().long()
        input_masks =  batch["input_masks"].cuda()
        input_segments = batch["input_segments"].cuda()
        extra_features = batch["extra_features"].cuda().float()
        labels = batch["labels"].cuda()
        logits =  model(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=input_segments,
            extra_features=extra_features,
        )
        return logits, labels

    elif num_bert == 2:
        q_input_ids = batch["q_input_ids"].cuda().long()
        q_input_masks =  batch["q_input_masks"].cuda()
        q_input_segments = batch["q_input_segments"].cuda()
        a_input_ids = batch["a_input_ids"].cuda().long()
        a_input_masks =  batch["a_input_masks"].cuda()
        a_input_segments = batch["a_input_segments"].cuda()        
        extra_features = batch["extra_features"].cuda().float()
        labels = batch["labels"].cuda()

        logits =  model(
            q_input_ids=q_input_ids,
            q_attention_mask=q_input_masks,
            q_token_type_ids=q_input_segments,
            a_input_ids=q_input_ids,
            a_attention_mask=q_input_masks,
            a_token_type_ids=q_input_segments,
            extra_features=extra_features,
        )

        return logits, labels
        # if return_logits:
        #     return criterion(logits, labels), logits
        # else:
        #     return criterion(logits, labels)




def train(config, model, criterion, optimizer, scheduler, loaders, fold, train_only_head=False):
    model.train()
    iteration = 0
    best_score = -1.0

    if train_only_head:
        num_epochs = config.train.warmup_num_epochs
    def is_backbone(n):
        return "bert" in n

    if train_only_head:
        for n, p in params:
            if is_backbone(n):
                p.requires_grad = False        
    else:
        num_epochs = config.train.num_epochs

    # if config.train.fp16:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

    for epoch in range(num_epochs):
        avg_loss = 0.0
        optimizer.zero_grad()
        for idx, batch in enumerate(tqdm(loaders['train'], desc="Train", ncols=80)):
            logits, labels = process_batch(config.model.num_bert, model, batch, criterion)
            loss = criterion(logits, labels)

            if config.train.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (iteration + 1) % config.train.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            iteration += 1

            avg_loss += loss.item() / (len(loaders['train']) * config.train.accumulation_steps)

        avg_val_loss, val_score, val_preds = evaluate(
            config, model, loaders['valid'], criterion)

        print(
            "Fold {}/{}: \t Epoch {}/{}: \t loss={:.4f} \t val_loss={:.4f} \t val_score={:.6f}".format(
                fold,
                config.data.num_folds,
                epoch + 1,
                num_epochs,
                avg_loss,
                avg_val_loss,
                val_score
            )
        )
        wandb.log({
            'train_loss':avg_loss,
            'valid_loss':avg_val_loss,
            'valid_score':val_score,
            })

        if val_score > best_score:
            best_score = val_score
            torch.save(
                model.state_dict(),
                config.work_dir + '/best_model_fold{}.pth'.format(fold),
            )
            np.save(config.work_dir + '/val_preds_fold{}'.format(fold), val_preds)


def evaluate(config, model, val_loader, criterion):
    avg_val_loss = 0.0
    model.eval()

    valid_preds = []
    original = []
    ids = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Valid", ncols=80)):
            logits, labels = process_batch(config.model.num_bert, model, batch, criterion)
            loss = criterion(logits, labels)

            avg_val_loss += loss.item() / len(val_loader)
            valid_preds.extend(logits.detach().cpu().numpy())
            original.extend(labels.detach().cpu().numpy())

        valid_preds = np.array(valid_preds)
        original = np.array(original)

        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()

        for i in range(len(config.data.output_columns)):
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)

    return avg_val_loss, score / len(config.data.output_columns), preds[np.argsort(ids)]


def infer(config, model, test_loader, test_shape):
    test_preds = np.zeros((test_shape, config.model.num_labels))
    model.eval()

    ids = []
    test_preds = []

    for idx, batch in enumerate(tqdm(test_loader, desc="Test", ncols=80)):
        with torch.no_grad():

            ids.extend(batch["idx"].cpu().numpy())

            predictions = model(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["input_masks"].cuda(),
                token_type_ids=batch["input_segments"].cuda(),
            )
            test_preds.extend(predictions.detach().cpu().numpy())

    test_preds = np.array(test_preds)

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output[np.argsort(ids)]
