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
from sklearn.preprocessing import MinMaxScaler

EXTRA_FEATURE_COLS = [
    'host_academia.stackexchange.com',
    'host_android.stackexchange.com',
    'host_anime.stackexchange.com',
    'host_apple.stackexchange.com',
    'host_askubuntu.com',
    'host_bicycles.stackexchange.com',
    'host_biology.stackexchange.com',
    'host_blender.stackexchange.com',
    'host_boardgames.stackexchange.com',
    'host_chemistry.stackexchange.com',
    'host_christianity.stackexchange.com',
    'host_codereview.stackexchange.com',
    'host_cooking.stackexchange.com',
    'host_crypto.stackexchange.com',
    'host_cs.stackexchange.com',
    'host_dba.stackexchange.com',
    'host_diy.stackexchange.com',
    'host_drupal.stackexchange.com',
    'host_dsp.stackexchange.com',
    'host_electronics.stackexchange.com',
    'host_ell.stackexchange.com',
    'host_english.stackexchange.com',
    'host_expressionengine.stackexchange.com',
    'host_gamedev.stackexchange.com',
    'host_gaming.stackexchange.com',
    'host_gis.stackexchange.com',
    'host_graphicdesign.stackexchange.com',
    'host_judaism.stackexchange.com',
    'host_magento.stackexchange.com',
    'host_math.stackexchange.com',
    'host_mathematica.stackexchange.com',
    'host_mathoverflow.net',
    'host_mechanics.stackexchange.com',
    'host_meta.askubuntu.com',
    'host_meta.christianity.stackexchange.com',
    'host_meta.codereview.stackexchange.com',
    'host_meta.math.stackexchange.com',
    'host_meta.stackexchange.com',
    'host_money.stackexchange.com',
    'host_movies.stackexchange.com',
    'host_music.stackexchange.com',
    'host_photo.stackexchange.com',
    'host_physics.stackexchange.com',
    'host_programmers.stackexchange.com',
    'host_raspberrypi.stackexchange.com',
    'host_robotics.stackexchange.com',
    'host_rpg.stackexchange.com',
    'host_salesforce.stackexchange.com',
    'host_scifi.stackexchange.com',
    'host_security.stackexchange.com',
    'host_serverfault.com',
    'host_sharepoint.stackexchange.com',
    'host_softwarerecs.stackexchange.com',
    'host_stackoverflow.com',
    'host_stats.stackexchange.com',
    'host_superuser.com',
    'host_tex.stackexchange.com',
    'host_travel.stackexchange.com',
    'host_unix.stackexchange.com',
    'host_ux.stackexchange.com',
    'host_webapps.stackexchange.com',
    'host_webmasters.stackexchange.com',
    'host_wordpress.stackexchange.com',
    'cat_CULTURE',
    'cat_LIFE_ARTS',
    'cat_SCIENCE',
    'cat_STACKOVERFLOW',
    'cat_TECHNOLOGY',
    'question_body_num_words',
    'answer_num_words',
    'question_vs_answer_length',
    'q_a_author_same',
    'answer_user_cat',
    'indirect',
    'question_count',
    'reason_explanation_words',
    'choice_words'
    ]

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) # + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments # + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids # + [0] * (max_seq_length - len(token_ids))
    return input_ids

def compute_output_arrays(df, columns, rank_transform=False):
    if rank_transform:
        # Min Max scale target after rank transformation
        for col in columns:
            df[col] = df[col].rank(method="average")
        df[columns] = MinMaxScaler().fit_transform(df[columns])

    label_weights = 1.0 / df[columns].std().values
    label_weights = label_weights / label_weights.sum() * 30
    np.save('data/label_weights', label_weights)

    return np.asarray(df[columns])

def get_pseudo_set(config, pseudo_df, tokenizer):
    return QuestDataset.from_frame(config, pseudo_df, tokenizer)


def get_test_set(config, test_df, tokenizer):
    return QuestDataset.from_frame(config, test_df, tokenizer, True)