import pandas as pd
import numpy as np
import os
import argparse
import warnings
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
warnings.filterwarnings('ignore')


def get_scaffold(smi):
    """Extract the Bemis-Murcko scaffold of a molecule; return the original SMILES as its own group if extraction fails"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else smi
    except Exception:
        return smi


def scaffold_10fold_split(data, smiles_col='smiles', n_splits=10, seed=2026):
    """
    Scaffold-based 10-fold cross validation split
    Ensures molecules with the same scaffold are in the same fold to prevent leakage into the training set
    seed =2022/2023/2024/2025/2026
    """
    np.random.seed(seed)
    
    data = data.reset_index(drop=True)
    data['scaffold'] = data[smiles_col].apply(get_scaffold)
    
    scaffold_groups = data.groupby('scaffold').groups
    scaffold_names = sorted(scaffold_groups.keys(), 
                            key=lambda x: len(scaffold_groups[x]), 
                            reverse=True)
    
    fold_sizes = [0] * n_splits
    scaffold_to_fold = {}
    
    for scf in scaffold_names:
        n_mol = len(scaffold_groups[scf])
        min_size = min(fold_sizes)
        min_folds = [i for i, size in enumerate(fold_sizes) if size == min_size]
        target_fold = int(np.random.choice(min_folds))
        scaffold_to_fold[scf] = target_fold
        fold_sizes[target_fold] += n_mol
    
    data['fold'] = data['scaffold'].map(scaffold_to_fold)
    
    fold_indices = []
    for fold_id in range(n_splits):
        test_idx = data[data['fold'] == fold_id].index.tolist()
        fold_indices.append(test_idx)
    
    print(f"Fold sample sizes: {[len(f) for f in fold_indices]}")
    print(f"Total unique scaffolds: {len(scaffold_groups)}")
    
    return fold_indices


def split_train_valid(data, train_idx, seed=2026, valid_ratio=0.1):
    """Split a validation set from the training set by scaffold, ensuring validation scaffolds do not leak into the training subset"""
    np.random.seed(seed)
    train_sub = data.iloc[train_idx].reset_index(drop=True)
    
    train_sub['scaffold'] = train_sub['smiles'].apply(get_scaffold)
    scaffolds = train_sub['scaffold'].unique()
    np.random.shuffle(scaffolds)
    
    val_scaffolds = set()
    val_count = 0
    target_val = int(len(train_sub) * valid_ratio)
    
    for scf in scaffolds:
        if val_count >= target_val:
            break
        scf_count = (train_sub['scaffold'] == scf).sum()
        val_scaffolds.add(scf)
        val_count += scf_count
    
    val_mask = train_sub['scaffold'].isin(val_scaffolds)
    valid_idx_local = train_sub[val_mask].index.tolist()
    train_idx_local = train_sub[~val_mask].index.tolist()
    
    final_train = [train_idx[i] for i in train_idx_local]
    final_valid = [train_idx[i] for i in valid_idx_local]
    
    return final_train, final_valid


def save_splits(data, fold_indices, output_dir, prefix='2026', seed=2026):
    os.makedirs(output_dir, exist_ok=True)
    
    for fold_id, test_idx in enumerate(fold_indices):
        train_idx = []
        for f in range(len(fold_indices)):
            if f != fold_id:
                train_idx.extend(fold_indices[f])
        
        final_train_idx, final_valid_idx = split_train_valid(data, train_idx, seed=seed+fold_id)
        
        data.iloc[final_train_idx].to_csv(
            os.path.join(output_dir, f'{prefix}_fold_{fold_id}_train.csv'), 
            index=False
        )
        data.iloc[final_valid_idx].to_csv(
            os.path.join(output_dir, f'{prefix}_fold_{fold_id}_valid.csv'), 
            index=False
        )
        data.iloc[test_idx].to_csv(
            os.path.join(output_dir, f'{prefix}_fold_{fold_id}_test.csv'), 
            index=False
        )
        
        total = len(data)
        print(f"Fold {fold_id:2d} | "
              f"Train: {len(final_train_idx):4d} ({len(final_train_idx)/total*100:.1f}%) | "
              f"Valid: {len(final_valid_idx):4d} ({len(final_valid_idx)/total*100:.1f}%) | "
              f"Test: {len(test_idx):4d} ({len(test_idx)/total*100:.1f}%)")
    
    print(f"\nAll splits saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Scaffold 10-fold split for molecular datasets')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='SMILES column name')
    parser.add_argument('--n_splits', type=int, default=10, help='Number of folds')
    parser.add_argument('--seed', type=int, nargs='+', default=[2026], help='Random seed(s), e.g., --seed 2022 2023 2024 2025 2026')
    parser.add_argument('--prefix', type=str, default=None, help='Output file prefix (if not specified, uses seed as prefix)')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    data = pd.read_csv(args.input)
    print(f"Total molecules: {len(data)}")
    
    seeds = args.seed
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Processing seed: {seed}")
        print(f"{'='*60}")
        
        print("\nGenerating scaffold-based 10-fold splits...")
        fold_indices = scaffold_10fold_split(
            data, 
            smiles_col=args.smiles_col, 
            n_splits=args.n_splits, 
            seed=seed
        )
        
        print("\nSplitting validation set and saving files...")
        current_prefix = str(seed) if args.prefix is None else args.prefix
        save_splits(data, fold_indices, args.output, prefix=current_prefix, seed=seed)
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {len(seeds)} seed(s): {seeds}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
