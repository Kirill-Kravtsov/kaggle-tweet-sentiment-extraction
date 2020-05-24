import argparse
import pandas as pd
from sklearn import model_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--df', type=str, default="data/train.csv")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.df)
    df = df.dropna().reset_index(drop=True)
    df["fold"] = -1

    kf = model_selection.StratifiedKFold(n_splits=args.num_folds)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):
        print(len(trn_), len(val_))
        df.loc[val_, 'fold'] = fold

    df.to_csv(f"data/train_stratified_{args.num_folds}folds.csv", index=False)
