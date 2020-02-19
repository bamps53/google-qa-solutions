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
    title,
    question,
    answer,
    max_sequence_length=290,
    t_max_len=30,
    q_max_len=128,
    a_max_len=128,
):
    # SICK THIS IS ALL SEEMS TO BE SICK

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError(
                "New sequence length should be %d, but is %d"
                % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4))
            )
        q_len_head = round(q_new_len / 2)
        q_len_tail = -1 * (q_new_len - q_len_head)
        a_len_head = round(a_new_len / 2)
        a_len_tail = -1 * (a_new_len - a_len_head)  ## Head+Tail method .
        t = t[:t_new_len]
        if config.data.head_tail:
            q = q[:q_len_head] + q[q_len_tail:]
            a = a[:a_len_head] + a[a_len_tail:]
        else:
            q = q[:q_new_len]
            a = a[:a_new_len]  ## No Head+Tail ,usual processing

    return t, q, a


def _convert_to_bert_inputs(
    title, question, answer, tokenizer, max_sequence_length
):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = (
        ["[CLS]"]
        + title
        + ["[SEP]"]
        + question
        + ["[SEP]"]
        + answer
        + ["[SEP]"]
    )

    return dict(
        input_ids = _get_ids(stoken, tokenizer, max_sequence_length),
        input_masks = _get_masks(stoken, max_sequence_length),
        input_segments = _get_segments(stoken, max_sequence_length),
        )


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
    all_inputs['input_ids'] = []
    all_inputs['input_masks'] = []
    all_inputs['input_segments'] = []
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
        t, q, a = _trim_input(
            config,
            tokenizer,
            t,
            q,
            a,
            max_sequence_length,
            t_max_len,
            q_max_len,
            a_max_len,
        )
        inputs = _convert_to_bert_inputs(
        #ids, masks, segments = _convert_to_bert_inputs(
            t, q, a, tokenizer, max_sequence_length
        )

    #     input_ids.append(np.array(ids, dtype=np.int64))
    #     input_masks.append(np.array(masks, dtype=np.int64))
    #     input_segments.append(np.array(segments, dtype=np.int64))

    # inputs = (
    #     input_ids,
    #     input_masks,
    #     input_segments,
    #     df[EXTRA_FEATURE_COLS].values
    # )

    # return inputs

        for k, v in all_inputs.items():
            v.append(np.array(inputs[k], dtype=np.int64))

    all_inputs['extra_features'] = df[EXTRA_FEATURE_COLS].values

    return all_inputs


class TQADataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None):
        self.inputs = inputs
        self.labels = labels
        self.lengths = lengths

    @classmethod
    def from_frame(cls, config, df, tokenizer, test=False):
        """ here I put major preprocessing. why not lol
        """
        inputs = compute_input_arays(config, df, tokenizer)

        outputs = None
        if not test:
            outputs = compute_output_arrays(df, config.data.output_columns, rank_transform=config.data.rank_transform)
            outputs = torch.tensor(outputs, dtype=torch.float32)

        lengths = [len(x) for x in inputs['input_ids']]

        return cls(inputs=inputs, lengths=lengths, labels=outputs)

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        sample = dict(
            idx=idx,
            input_ids =        self.inputs['input_ids'][idx],
            input_masks =      self.inputs['input_masks'][idx],
            input_segments =   self.inputs['input_segments'][idx],
            extra_features =     self.inputs['extra_features'][idx],
            lengths =            self.lengths[idx]
        )
        if self.labels is not None:
            sample["labels"] = self.labels[idx]

        return sample


    #     lengths = [len(x) for x in inputs[0]]

    #     return cls(inputs=inputs, lengths=lengths, labels=outputs)

    # def __len__(self):
    #     return len(self.inputs[0])

    # def __getitem__(self, idx):
    #     sample = dict(
    #         idx=idx,
    #         input_ids=self.inputs[0][idx],
    #         input_masks=self.inputs[1][idx],
    #         input_segments=self.inputs[2][idx],
    #         extra_features=self.inputs[3][idx],
    #         lengths=self.lengths[idx]
    #     )
    #     if self.labels is not None:
    #         sample["labels"] = self.labels[idx]

    #     return sample