import pandas as pd


def process_text(text):
    text = " ".join(text.lower().split()).strip()
    return text


data = pd.read_csv("../data/external/sentiment140.csv",
        encoding = "cp1252", header=None)

texts = list(data.iloc[:, 5].astype(str).apply(process_text))
long_str = "\n".join(texts)
with open("../data/external/sentiment140_lm.txt", "w+") as f:
    f.write(long_str)
