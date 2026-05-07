import pandas as pd
from sklearn.model_selection import GroupKFold
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base_dir, 'data_curated.csv'))


kf = GroupKFold(n_splits=10)
groups = data['scaffold'].values

num = 0

for train_idx, test_idx in kf.split(data, groups=groups):
    train_set = data.iloc[train_idx]
    test_set = data.iloc[test_idx]

    train_set.to_csv(os.path.join(base_dir, "2026_fold_{}_train.csv".format(num)), index=None)
    test_set.to_csv(os.path.join(base_dir, "2026_fold_{}_test.csv".format(num)), index=None)
    val_set = train_set.sample(frac=0.1, random_state=42)
    train_set = train_set.drop(val_set.index)
    val_set.to_csv(os.path.join(base_dir, "2026_fold_{}_valid.csv".format(num)), index=None)

    num = num + 1