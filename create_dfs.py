import numpy as np
import pandas as pd
from math import floor, ceil
from nltk.tokenize.treebank import TreebankWordTokenizer
from spacy.lang.en import English
from tqdm import tqdm
from config.base import load_config
from pathlib import Path
import argparse
from sklearn.preprocessing import MinMaxScaler
import os
from utils import INPUT_COLS
tqdm.pandas()
nlp = English()
sentencizer = nlp.create_pipe('sentencizer')
nlp.add_pipe(sentencizer)
from sklearn.model_selection import GroupKFold, KFold

def get_tree_tokens(x):
    tree_tokenizer = TreebankWordTokenizer()
    x = tree_tokenizer.tokenize(x)
    x = ' '.join(x)
    return x


def split_document(texts):
    all_sents = []
    max_num_sentences = 0.0
    for text in texts:
        doc = nlp(text)
        sents=[]
        for i,sent in enumerate(doc.sents):
            sents.append(sent.text)
        all_sents.append(sents)
    return all_sents


def add_question_metadata_features(text):
    doc=nlp(text)
    indirect = 0
    choice_words=0
    reason_explanation_words = 0
    question_count = 0

    for sent in doc.sents:
        if '?' in sent.text and '?' == sent.text[-1]:
            question_count += 1
            for token in sent:
                if token.text.lower()=='why':
                    reason_explanation_words+=1
                elif token.text.lower()=='or':
                    choice_words+=1
    if question_count==0:
        indirect+=1

    return np.array([indirect, question_count, reason_explanation_words, choice_words])

def question_answer_author_same(df):
    q_username = df['question_user_name']
    a_username = df['answer_user_name']
    author_same=[]
    
    for i in range(len(df)):
        if q_username[i] == a_username[i]:
            author_same.append(int(1))
        else:
            author_same.append(int(0))
    return author_same
        
    
def add_external_features(df, ans_user_and_category=None):
    df['question_body'] = df['question_body'].progress_apply(lambda x: str(x))
    df['question_body_num_words'] = df['question_body'].str.count('\S+')

    df['answer'] = df['answer'].progress_apply(lambda x: str(x))
    df['answer_num_words'] = df['answer'].str.count('\S+')

    df['question_vs_answer_length'] = df['question_body_num_words']/df['answer_num_words']

    df['q_a_author_same'] = question_answer_author_same(df)

    if ans_user_and_category is None:
        ans_user_and_category = \
            df[df[['answer_user_name', 'category']].duplicated()][['answer_user_name', 'category']].values
        return_feature = True
    else:
        return_feature = False

    answer_user_cat = []
    for i in tqdm(df[['answer_user_name', 'category']].values):
        if i in ans_user_and_category:
            answer_user_cat.append(int(1))
        else:
            answer_user_cat.append(int(0))
    df['answer_user_cat'] = answer_user_cat

    handmade_features=[]
    for text in df['question_body'].values:
        handmade_features.append(add_question_metadata_features(text))

    df = pd.concat([
        df,
        pd.DataFrame(
            handmade_features,
            columns=['indirect', 'question_count', 'reason_explanation_words', 'choice_words']
            )],
        axis=1
        )
    if return_feature:
        return df, ans_user_and_category
    else:
        return df

def convert_cotegorical(df, col, prefix, categories=None):
    if categories == None:
        categories = set(df[col].unique().tolist())
        return_cat = True
    else:
        return_cat = False

    df[col] = pd.Categorical(df[col], categories=categories)
    df = pd.get_dummies(df, columns=[col], drop_first=False, prefix=prefix)

    if return_cat:
        return df, categories
    else:
        return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', default=5, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    num_words_scaler = MinMaxScaler()

    df_train = pd.read_csv('../input/google-quest-challenge/train.csv',header=0,encoding='utf-8')
    # for col in INPUT_COLS:
    #     df_train[col] = df_train[col].apply(lambda x: get_tree_tokens(x))
    df_train, ans_user_and_category = add_external_features(df_train)
    df_train[['question_body_num_words', 'answer_num_words']] = \
        num_words_scaler.fit_transform(df_train[['question_body_num_words', 'answer_num_words']].values)
    df_train, host_categories = convert_cotegorical(df_train, col='host', prefix='host')
    df_train, cat_categories = convert_cotegorical(df_train, col='category', prefix='cat')

    df_train['fold'] = -1
    kf = GroupKFold(n_splits=args.num_folds)
    for fold, (_, val_idx) in enumerate(kf.split(df_train.values, groups=df_train.question_title)):
        df_train.loc[val_idx, 'fold'] = fold

    df_test = pd.read_csv('../input/google-quest-challenge/test.csv',header=0,encoding='utf-8')
    # for col in INPUT_COLS:
    #     df_test[col] = df_test[col].apply(lambda x: get_tree_tokens(x))
    df_test = add_external_features(df_test, ans_user_and_category)
    df_test[['question_body_num_words', 'answer_num_words']] = \
        num_words_scaler.transform(df_test[['question_body_num_words', 'answer_num_words']].values)
    df_test = convert_cotegorical(df_test, col='host', prefix='host', categories=host_categories)
    df_test = convert_cotegorical(df_test, col='category', prefix='cat', categories=cat_categories)

    os.makedirs('data/', exist_ok=True)
    df_train.to_csv('data/train_df.csv', index=False)
    df_test.to_csv('data/test_df.csv', index=False)