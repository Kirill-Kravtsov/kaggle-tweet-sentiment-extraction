import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    df = df.dropna().reset_index(drop=True)
    df["fold"] = -1

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):
        print(len(trn_), len(val_))
        df.loc[val_, 'fold'] = fold

    df.to_csv("data/train_stratified_5folds_new2.csv", index=False)
