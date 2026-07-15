"""
Random Split Script

Split logic:
- 10-fold cross validation
- In each fold, 1 fold as independent test set (~10%)
- Remaining 9 folds as train+validation pool (~90%)
- Split the 9 folds into train set (~81%) and validation set (~9%) at 9:1 ratio
- Final ratio: train 81% : validation 9% : test 10%
- Validation set is only used for saving the best checkpoint, not for test set evaluation
- Supports multiple seed splits: [2022, 2023, 2024, 2025, 2026]
"""

import pandas as pd
from sklearn.model_selection import KFold
import os
import argparse

DEFAULT_SEEDS = [2022, 2023, 2024, 2025, 2026]

def split_dataset(data_path, output_dir, seeds=DEFAULT_SEEDS):
    """
    Perform modified 10-fold cross validation split on the dataset
    
    Args:
        data_path: Path to the original data file
        output_dir: Output directory
        seeds: Random seed list, default [2022, 2023, 2024, 2025, 2026]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    print(f"Original dataset size: {len(data)}")
    print(f"Using seed list: {seeds}")
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Processing seed: {seed}")
        print(f"{'='*50}")
        
        kf_outer = KFold(n_splits=10, shuffle=True, random_state=seed)
        
        fold_num = 0
        
        for train_val_idx, test_idx in kf_outer.split(data):
            train_val_set = data.iloc[train_val_idx]
            test_set = data.iloc[test_idx]
            
            kf_inner = KFold(n_splits=10, shuffle=True, random_state=seed)
            
            inner_splits = list(kf_inner.split(train_val_set))
            train_idx_inner, val_idx_inner = inner_splits[0]
            
            train_set = train_val_set.iloc[train_idx_inner]
            val_set = train_val_set.iloc[val_idx_inner]
            
            train_ratio = len(train_set) / len(data) * 100
            val_ratio = len(val_set) / len(data) * 100
            test_ratio = len(test_set) / len(data) * 100
            
            print(f"\nFold {fold_num}:")
            print(f"  Train set: {len(train_set)} ({train_ratio:.1f}%)")
            print(f"  Validation set: {len(val_set)} ({val_ratio:.1f}%)")
            print(f"  Test set: {len(test_set)} ({test_ratio:.1f}%)")
            print(f"  Validation and test sets overlap: {len(set(val_idx_inner) & set(test_idx)) == 0}")
            
            train_set.to_csv(os.path.join(output_dir, f"{seed}_fold_{fold_num}_train.csv"), index=None)
            val_set.to_csv(os.path.join(output_dir, f"{seed}_fold_{fold_num}_valid.csv"), index=None)
            test_set.to_csv(os.path.join(output_dir, f"{seed}_fold_{fold_num}_test.csv"), index=None)
            
            fold_num += 1
        
        print(f"\nSeed {seed} split complete! Total {fold_num} folds")
    
    print(f"\n{'='*50}")
    print(f"All seed splits complete!")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Modified dataset random split')
    parser.add_argument('--data', type=str, required=True, help='Path to the original data file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS, 
                        help=f'Random seed list, default {DEFAULT_SEEDS}')
    
    args = parser.parse_args()
    
    split_dataset(args.data, args.output, args.seeds)

if __name__ == '__main__':
    main()