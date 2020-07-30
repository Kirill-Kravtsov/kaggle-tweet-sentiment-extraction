import re
import random
import pydoc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_utils import find_substr
from nltk.tokenize.casual import WORD_RE, HANG_RE, _replace_html_entities


class TweetDataset(Dataset):

    def __init__(self, df_path, folds, tokenizer,
                 preprocess_fn="datasets.roberta_preprocess",
                 max_len=192, max_num_samples=None,
                 text_col_name="text", do_lower=True, **kwargs):
        if isinstance(folds, int):
            folds = [folds]
        df = pd.read_csv(df_path)
        df = df[df['fold'].isin(folds)]
        if max_num_samples is not None:
            df = df.iloc[:max_num_samples]
        self.tweet = df[text_col_name].values
        self.sentiment = df['sentiment'].values
        self.selected_text = df['selected_text'].values
        self.tweet_ids = df['textID'].values
        self.fn = pydoc.locate(preprocess_fn)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.do_lower = do_lower
        self.kwargs = kwargs

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, idx):
        return self.fn(
            self.tweet[idx],
            self.selected_text[idx],
            self.sentiment[idx],
            self.tweet_ids[idx],
            self.tokenizer,
            self.max_len,
            self.do_lower,
            **self.kwargs
        )


def swap_words(text):
    words = text.split()
    if len(words)<=1:
        return text
    swap_idx1 = random.randint(0, len(words)-2)
    swap_idx2 = swap_idx1 + 1  # swap with the next one
    tmp = words[swap_idx1]
    words[swap_idx1] = words[swap_idx2]
    words[swap_idx2] = tmp
    return " ".join(words)


def get_new_words_vect(offsets, tweet):
    new_words = []
    for off_from, off_to in offsets:
        new_words.append(0 if off_from == 0 else tweet[off_from-1] == " ")
    return new_words


def roberta_preprocess(tweet, selected_text, sentiment, tweet_id, tokenizer, max_len,
                       do_lower=True, p_swap_words=0, is_valid_df=False):
    tweet = " ".join(str(tweet).strip().split())
    selected_text = " ".join(str(selected_text).strip().split())

    len_st = len(selected_text)
    idx0, idx1 = find_substr(tweet, selected_text)
    assert selected_text == tweet[idx0:idx1+1]

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1

    if (not is_valid_df) and (random.random() < p_swap_words):
        tweet = tweet[:idx1+1] + swap_words(tweet[idx1+1:])
    #if (not is_valid_df) and (random.random() < p_swap_words):
    #    tweet = swap_words(tweet[:idx0]) + tweet[idx0:]

    if do_lower:
        tok_tweet = tokenizer.encode_plus(tweet.lower(), return_offsets_mapping=True, add_special_tokens=False)
    else:
        tok_tweet = tokenizer.encode_plus(tweet, return_offsets_mapping=True, add_special_tokens=False)
    input_ids_orig = tok_tweet['input_ids']
    tweet_offsets = tok_tweet['offset_mapping']
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    new_words = get_new_words_vect(tweet_offsets, tweet)
    bin_sentiment = np.zeros(len(input_ids))
    bin_sentiment[targets_start:targets_end+1] = 1
    bin_sentiment_words = bin_sentiment * np.array(new_words)

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        new_words = new_words + ([0] * padding_length)
        bin_sentiment = bin_sentiment.tolist() + ([0] * padding_length)
        bin_sentiment_words  = bin_sentiment_words.tolist() + ([0] * padding_length)

    return {
        'tweet_id': tweet_id,
        'ids': torch.tensor(input_ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'start_positions': torch.tensor(targets_start, dtype=torch.long),
        'end_positions': torch.tensor(targets_end, dtype=torch.long),
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': torch.tensor(tweet_offsets, dtype=torch.long),
        'new_words': torch.tensor(new_words, dtype=torch.float),
        'bin_sentiment': torch.tensor(bin_sentiment, dtype=torch.float),
        'bin_sentiment_words': torch.tensor(bin_sentiment_words, dtype=torch.float)
    }


def gpt2_preprocess(tweet, selected_text, sentiment, tokenizer, max_len,
                       do_lower=True):
    tweet = " ".join(str(tweet).strip().split())
    selected_text = " ".join(str(selected_text).strip().split())

    len_st = len(selected_text)
    idx0, idx1 = find_substr(tweet, selected_text)
    assert selected_text == tweet[idx0:idx1+1]

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1

    if do_lower:
        tok_tweet = tokenizer.encode_plus(tweet.lower(), return_offsets_mapping=True, add_special_tokens=False)
    else:
        tok_tweet = tokenizer.encode_plus(tweet, return_offsets_mapping=True, add_special_tokens=False)
    input_ids_orig = tok_tweet['input_ids']
    tweet_offsets = tok_tweet['offset_mapping']

    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    pad_id = tokenizer.pad_token_id
    if padding_length > 0:
        input_ids = input_ids + ([pad_id] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

    return {
        'ids': torch.tensor(input_ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'start_positions': torch.tensor(targets_start, dtype=torch.long),
        'end_positions': torch.tensor(targets_end, dtype=torch.long),
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': torch.tensor(tweet_offsets, dtype=torch.long)
    }



def custom_tweet_tokenize(text):
    return WORD_RE.findall(text)


def normalize_token(token):
    return token


def normalize_space_text(text):
    """Jaccard-safe whitespace normalization"""
    return " ".join(text.strip().split())


def process_tweet(tweet, selected_text):
    """
    Perform normalization and create mapping from
    processed to original
    """
    tokenized = custom_tweet_tokenize(tweet)
    selected_text = " ".join(map(normalize_token, custom_tweet_tokenize(selected_text)))

    idx_orig = 0
    idx_processed = 1  # because of leading space
    processed2orig_idx = {}
    tokenized_normalized = []

    for token in tokenized:
        # find current token
        idx_orig_from = tweet.find(token, idx_orig)
        idx_orig_to = idx_orig_from + len(token) - 1  # inclusive
        # here could be some token transformation
        token = normalize_token(token)
        tokenized_normalized.append(token)

        idx_processed_from = idx_processed
        idx_processed_to = idx_processed_from + len(token) - 1  # inclusive

        # reassign global pointers
        idx_orig = idx_orig_to + 1
        idx_processed = idx_processed_to + 2  # skip next space

        # fill the mapping dict
        for map_idx_processed, map_idx_orig in zip(range(idx_processed_from, idx_processed_to+1),
                                                   range(idx_orig_from, idx_orig_to+1)):
            processed2orig_idx[map_idx_processed] = map_idx_orig

        # if word's length was changed during normalization
        #   match at least ending indexes
        processed2orig_idx[idx_processed_to] = idx_orig_to

    tweet = " " + " ".join(tokenized_normalized)
    return tweet, selected_text, processed2orig_idx



def bertweet_preprocess(tweet, selected_text, sentiment, tweet_id, tokenizer, max_len, do_lower=True, is_valid_df=False):
    tweet = normalize_space_text(str(tweet))
    selected_text = normalize_space_text(str(selected_text))

    tweet_norm, selected_norm, norm2orig = process_tweet(tweet, selected_text)
    len_st = len(selected_norm)

    idx = tweet_norm.find(selected_norm)
    char_targets = np.zeros((len(tweet_norm)))
    char_targets[idx:idx+len(selected_norm)]=1
    tok_tweet = tokenizer.encode(tweet_norm)

    # ID_OFFSETS
    splitted = tokenizer.bpe_encode(tweet_norm).split()
    offsets = []; idx=0
    for w in splitted:
        if (len(w) >= 2) and (w[-2:]=="@@"):
            w = w[:-2]
        if tweet_norm[tweet_norm.find(w, idx)-1] == " ":
            idx += 1
            offsets.append((idx, idx+len(w)))
            idx += len(w)
        else:
            offsets.append((idx, idx+len(w)))
            idx += len(w)

    input_ids_orig = tok_tweet
    tweet_offsets = offsets

    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if np.sum(char_targets[offset1:offset2])> 0:
            target_idx.append(j)

    if  len(target_idx)>0:
        targets_start = target_idx[0]
        targets_end = target_idx[-1]
    else:
        targets_start = 0
        targets_end= len(char_targets)

    fixed_offsets = []
    for off_from, off_to in tweet_offsets:
        off_from_fixed = norm2orig[off_from]
        off_to_fixed = norm2orig[off_to-1] + 1
        fixed_offsets.append((off_from_fixed, off_to_fixed))

    sentiment_id = {
        'positive': 1809,
        'negative': 3392,
        'neutral': 14058
    }

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    fixed_offsets = [(0, 0)] * 4 + fixed_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        fixed_offsets = fixed_offsets + ([(0, 0)] * padding_length)

    return {
        'tweet_id': tweet_id,
        'ids': torch.tensor(input_ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'start_positions': torch.tensor(targets_start, dtype=torch.long),
        'end_positions': torch.tensor(targets_end, dtype=torch.long),
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': torch.tensor(fixed_offsets, dtype=torch.long)
    }
