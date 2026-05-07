import pandas as pd
from sklearn.model_selection import KFold
import os

base_dir=os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base_dir,'data_curated.csv'))

kf = KFold(n_splits=10)

num = 0

for train,test in kf.split(data):
    train_set = data.iloc[train]
    test_set = data.iloc[test]
    train_set.to_csv(os.path.join(base_dir,"2026_fold_{}_train.csv".format(num)),index=None)
    test_set.to_csv(os.path.join(base_dir,"2026_fold_{}_test.csv".format(num)),index=None)
    test_set.to_csv(os.path.join(base_dir,"2026_fold_{}_valid.csv".format(num)),index=None)
    num = num+1