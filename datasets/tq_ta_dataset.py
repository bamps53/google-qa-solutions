from math import floor, ceil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .common import _get_ids, _get_masks, _get_segments, EXTRA_FEATURE_COLS, compute_output_arrays

def _trim_input(
    config,
    tokenizer,
    text1,
    text2,
    max_sequence_length=512,
    t1_max_len=50,
    t2_max_len=459,
):
    # SICK THIS IS ALL SEEMS TO BE SICK

    t1 = tokenizer.tokenize(text1)
    t2 = tokenizer.tokenize(text2)

    t1_len = len(t1)
    t2_len = len(t2)

    if (t1_len + t2_len + 3) > max_sequence_length:

        if t1_max_len > t1_len:
            t1_new_len = t1_len
            t2_new_len = max_sequence_length - t1_new_len - 3
        else:
            t2_new_len = t2_len
            t1_new_len = max_sequence_length - t2_new_len - 3

        if t1_new_len + t2_new_len + 3 != max_sequence_length:
            raise ValueError(
                "New sequence length should be %d, but is %d"
                % (max_sequence_length, (t1_new_len + t2_new_len + 3))
            )

        t1_len_head = round(t1_new_len / 2)
        t1_len_tail = -1 * (t1_new_len - t1_len_head)
        t2_len_head = round(t2_new_len / 2)
        t2_len_tail = -1 * (t2_new_len - t2_len_head)  ## Head+Tail method .

        if config.data.head_tail:
            t1 = t1[:t1_len_head] + t1[t1_len_tail:]
            t2 = t2[:t2_len_head] + t2[t2_len_tail:]
        else:
            t1 = t1[:t1_new_len]
            t2 = t2[:t2_new_len]  ## No Head+Tail ,usual processing

    return t1, t2


def _convert_to_bert_inputs(
    t1, q, t2, a, tokenizer, max_sequence_length
):
    """Converts tokenized input to ids, masks and segments for BERT"""

    q_stoken = (
        ["[CLS]"]
        + t1
        + ["[SEP]"]
        + q
        + ["[SEP]"]
    )

    a_stoken = (
        ["[CLS]"]
        + t2
        + ["[SEP]"]
        + a
        + ["[SEP]"]
    )

    q_input_ids = _get_ids(q_stoken, tokenizer, max_sequence_length)
    q_input_masks = _get_masks(q_stoken, max_sequence_length)
    q_input_segments = _get_segments(q_stoken, max_sequence_length)

    a_input_ids = _get_ids(a_stoken, tokenizer, max_sequence_length)
    a_input_masks = _get_masks(a_stoken, max_sequence_length)
    a_input_segments = _get_segments(a_stoken, max_sequence_length)    

    inputs = {}
    inputs['q_input_ids'] = q_input_ids
    inputs['q_attention_mask'] = q_input_masks
    inputs['q_input_segments'] = q_input_segments
    inputs['a_input_ids'] = a_input_ids
    inputs['a_attention_mask'] = a_input_masks
    inputs['a_input_segments'] = a_input_segments
    return inputs

def compute_input_arays(
    config,
    df,
    tokenizer,
):
    columns = config.data.input_columns
    max_sequence_length=config.data.max_sequence_length
    t_max_len=config.data.max_title_length
    q_max_len=config.data.max_question_length
    a_max_len=config.data.max_answer_length
    all_inputs = {}
    all_inputs['q_input_ids'] = []
    all_inputs['q_attention_mask'] = []
    all_inputs['q_input_segments'] = []
    all_inputs['a_input_ids'] = []
    all_inputs['a_attention_mask'] = []
    all_inputs['a_input_segments'] = []
    for _, instance in tqdm(
        df[columns].iterrows(),
        desc="Preparing dataset",
        total=len(df),
        ncols=80,
    ):
        t, q, a = (
            instance.question_title,
            instance.question_body,
            instance.answer,
        )

        t1, q = _trim_input(
            config,
            tokenizer,
            t,
            q,
            max_sequence_length,
            t_max_len,
            q_max_len,
        )

        t2, a = _trim_input(
            config,
            tokenizer,
            t,
            a,
            max_sequence_length,
            t_max_len,
            a_max_len,
        )
        inputs = _convert_to_bert_inputs(
            t1, q, t2, a, tokenizer, max_sequence_length
        )
        for k, v in all_inputs.items():
            v.append(np.array(inputs[k], dtype=np.int64))

    all_inputs['extra_features'] = df[EXTRA_FEATURE_COLS].values

    return all_inputs


class TQTADataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None, cutout=0):
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths
        self.cutout = cutout

    @classmethod
    def from_frame(cls, config, df, tokenizer, test=False, cutout=0):
        """ here I put major preprocessing. why not lol
        """
        inputs = compute_input_arays(config, df, tokenizer)

        outputs = None
        if not test:
            outputs = compute_output_arrays(df, config.data.output_columns, rank_transform=config.data.rank_transform)
            outputs = torch.tensor(outputs, dtype=torch.float32)

        lengths = [len(x) for x in inputs['q_input_ids']]

        return cls(inputs=inputs, lengths=lengths, labels=outputs, cutout=cutout)

    def __len__(self):
        return len(self.inputs['q_input_ids'])

    # def __getitem__(self, idx):
    #     sample = dict(
    #         idx=idx,
    #         q_input_ids =        self.inputs['q_input_ids'][idx],
    #         q_input_masks =      self.inputs['q_input_masks'][idx],
    #         q_input_segments =   self.inputs['q_input_segments'][idx],
    #         a_input_ids =        self.inputs['a_input_ids'][idx],
    #         a_input_masks =      self.inputs['a_input_masks'][idx],
    #         a_input_segments =   self.inputs['a_input_segments'][idx],
    #         extra_features =     self.inputs['extra_features'][idx],
    #         lengths =            self.lengths[idx]
    #     )
    #     if self.labels is not None:
    #         sample["labels"] = self.labels[idx]

    #     return sample

    def __getitem__(self, idx):
        inputs = dict(
            # idx=idx,
            q_input_ids =        self.inputs['q_input_ids'][idx],
            q_attention_mask =      self.inputs['q_attention_mask'][idx],
            q_input_segments =   self.inputs['q_input_segments'][idx],
            a_input_ids =        self.inputs['a_input_ids'][idx],
            a_attention_mask =      self.inputs['a_attention_mask'][idx],
            a_input_segments =   self.inputs['a_input_segments'][idx],
            extra_features =     self.inputs['extra_features'][idx],
            # lengths =            self.lengths[idx]
        )

        if self.cutout > 0:
            q_len = len(inputs['q_input_ids'])
            a_len = len(inputs['a_input_ids'])
            q_cutout_len = int(q_len * self.cutout)
            a_cutout_len = int(a_len * self.cutout)
            q_start = np.random.randint(q_len - q_cutout_len)
            a_start = np.random.randint(a_len - a_cutout_len)

            inputs['q_input_ids'][q_start:q_start+q_cutout_len] = 0
            inputs['q_attention_mask'][q_start:q_start+q_cutout_len] = 0
            # inputs['q_input_segments'][q_start:q_start+q_cutout_len] = 0
            inputs['a_input_ids'][a_start:a_start+a_cutout_len] = 0
            inputs['a_attention_mask'][a_start:a_start+a_cutout_len] = 0
            # inputs['a_input_segments'][a_start:a_start+a_cutout_len] = 0

        if self.labels is not None:
            labels = self.labels[idx]

        return inputs, labels

