from math import floor, ceil

import torch
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
    MultilabelStratifiedKFold,
)
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import GroupKFold, KFold
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from .tq_ta_dataset import TQTADataset
from .tqa_dataset import TQADataset

class BucketingSampler:

    def __init__(self, lengths, batch_size, maxlen=500):

        self.lengths = lengths
        self.batch_size = batch_size
        self.maxlen = 500

        self.batches = self._make_batches(lengths, batch_size, maxlen)

    def _make_batches(self, lengths, batch_size, maxlen):

        max_total_length = maxlen * batch_size
        ids = np.argsort(lengths)

        current_maxlen = 0
        batch = []
        batches = []

        for id in ids:
            current_len = len(batch) * current_maxlen
            size = lengths[id]
            current_maxlen = max(size, current_maxlen)
            new_len = current_maxlen * (len(batch) + 1)
            if new_len < max_total_length:
                batch.append(id)
            else:
                batches.append(batch)
                current_maxlen = size
                batch = [id]

        if batch:
            batches.append(batch)

        assert (sum(len(batch) for batch in batches)) == len(lengths)

        return batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def make_collate_fn(padding_values):

    def _collate_fn(batch):
        for name, padding_value in padding_values.items():

            lengths = [len(sample[0][name]) for sample in batch]
            max_length = max(lengths)

            for n, size in enumerate(lengths):
                p = max_length - size
                if p:
                    pad_width = [(0, p)] + [(0, 0)] * (batch[n][0][name].ndim - 1)
                    if padding_value == "edge":
                        batch[n][0][name] = np.pad(
                            batch[n][0][name], pad_width,
                            mode="edge")
                    else:
                        batch[n][0][name] = np.pad(
                            batch[n][0][name], pad_width,
                            mode="constant", constant_values=padding_value)
        return default_collate(batch)

    return _collate_fn


def get_train_val_loaders(config, idx_fold, debug):
    train_df = pd.read_csv(config.data.train_df_path)
    trn_df = train_df[train_df['fold'] != idx_fold]
    val_df = train_df[train_df['fold'] == idx_fold]
    if debug:
        trn_df = trn_df.iloc[:100]
        val_df = val_df.iloc[:100]

    tokenizer = BertTokenizer.from_pretrained(
        config.model.name, do_lower_case=("uncased" in config.model.name)
    )

    if config.model.num_bert == 1:
        train_set = TQADataset.from_frame(config, trn_df, tokenizer)
        valid_set = TQADataset.from_frame(config, val_df, tokenizer)
        padding_values = {
            "input_ids": 0,
            "input_masks": 0,
            "input_segments": 0,
            }
    elif config.model.num_bert == 2:
        train_set = TQTADataset.from_frame(config, trn_df, tokenizer, cutout=config.data.cutout)
        valid_set = TQTADataset.from_frame(config, val_df, tokenizer, cutout=0)
        padding_values = {
            "q_input_ids": 0,
            "q_attention_mask": 0,
            "q_input_segments": 0,
            "a_input_ids": 0,
            "a_attention_mask": 0,
            "a_input_segments": 0,
            }        


    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        num_workers=0,
        collate_fn=make_collate_fn(padding_values=padding_values),
        drop_last=True,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_sampler=BucketingSampler(
            valid_set.lengths,
            batch_size=config.test.batch_size,
            maxlen=config.data.max_sequence_length
        ),
        collate_fn=make_collate_fn(padding_values=padding_values),
    )

    return dict(train=train_loader, valid=valid_loader)