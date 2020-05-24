import pandas as pd


def process_text(text):
    text = " ".join(text.lower().split()).strip()
    return text


train = pd.read_csv("../data/train.csv").dropna()
test = pd.read_csv("../data/test.csv")

texts = list(train['text'].astype(str).apply(process_text)) + list(test['text'].astype(str).apply(process_text))
long_str = "\n".join(texts)
with open("../data/generated/train_test.txt", "w+") as f:
    f.write(long_str)
