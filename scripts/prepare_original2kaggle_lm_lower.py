import sys
sys.path.append("../src")
import re
import pandas as pd
from data_utils import find_substr
from better_profanity import profanity

profanity.load_censor_words(whitelist_words=['asses'])
profanity.add_censor_words(['gay'])


def symb_decode(s):
    replacements = (
            ('&#39;', "'"),
            ("'", "`"),
            ('&quot;', "'"),
            ('&gt;', '>'),
            ('&lt;', '<'),
            ('&amp;', '&'),
        )
    for replace in replacements:
        s = s.replace(replace[0], replace[1])
    return s


def filter_name(text):
    return re.sub('@[a-zA-Z0-9]+', '', text)


def filter_name_correct(text):
    return re.sub('@[(a-zA-Z0-9)|_]+', '', text)


def kaggle_process(text):
    return profanity.censor(filter_name(symb_decode(text)))


def process_text_lm(text):
    text = " ".join(text.lower().split()).strip()
    return text


original = pd.read_csv("../data/original.csv")
original['text_processed'] = original['content'].apply(kaggle_process)
original['text_processed'] = original['text_processed'].apply(process_text_lm)

texts = list(original['text_processed'])
long_str = "\n".join(texts)

with open("../data/external/original_with_kaggle_preprocess_lower.txt", "w+") as f:
    f.write(long_str)
