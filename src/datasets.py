import re
import pydoc
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_utils import find_substr


def roberta_preprocess(tweet, selected_text, sentiment, tokenizer, max_len,
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
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
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


def roberta_preprocess2(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " ".join(str(tweet).strip().split())
    selected_text = " ".join(str(selected_text).strip().split())

    len_st = len(selected_text)
    idx0, idx1 = find_substr(tweet, selected_text)
    assert selected_text == tweet[idx0:idx1+1]

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode_plus(tweet.lower(), return_offsets_mapping=True, add_special_tokens=False)
    input_ids_orig = tok_tweet['input_ids']
    tweet_offsets = tok_tweet['offset_mapping']
    
    target = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target.append(1)
        else:
            target.append(0)
    
    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    target = [0] * 4 + target + [0]

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        target = target + ([0] * padding_length)
    
    return {
        'ids': torch.tensor(input_ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'targets': torch.tensor(target, dtype=torch.float),
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': torch.tensor(tweet_offsets, dtype=torch.long)
    }


class TweetDataset(Dataset):

    def __init__(self, df_path, folds, tokenizer,
                 preprocess_fn="datasets.roberta_preprocess",
                 max_len=192, max_num_samples=None,
                 text_col_name="text", do_lower=True):
        if isinstance(folds, int):
            folds = [folds]
        df = pd.read_csv(df_path)
        df = df[df['fold'].isin(folds)]
        if max_num_samples is not None:
            df = df.iloc[:max_num_samples]
        self.tweet = df[text_col_name].values
        self.sentiment = df['sentiment'].values
        self.selected_text = df['selected_text'].values
        self.fn = pydoc.locate(preprocess_fn)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.do_lower = do_lower
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        return self.fn(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len,
            self.do_lower
        )
