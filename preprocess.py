import scanpy as sc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import scipy.sparse
import os
from sklearn.preprocessing import StandardScaler

NUM_CLASSES = 3

class GeneExpressionDataset(Dataset):
    def __init__(self, expressions, cell_types):
        normalized_expressions = self.normalize_data(expressions)
        self.expressions = torch.FloatTensor(normalized_expressions)
        # One-hot encode cell types
        unique_types = sorted(list(set(cell_types)))
        self.type_to_idx = {t: i for i, t in enumerate(unique_types)}
        self.cell_types = torch.LongTensor([self.type_to_idx[t] for t in cell_types])
        
    def __len__(self):
        return len(self.expressions)
    
    def __getitem__(self, idx):
        return self.expressions[idx], self.cell_types[idx]
    
    def normalize_data(self, data):
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)
        
        return normalized


def preprocess_data(input_file, output_dir, cell_types=['Monocyte', 'CD4 T cell', 'NK cell']):
    os.makedirs(output_dir, exist_ok=True)
    adata = sc.read_h5ad(input_file)
    adata = adata[adata.obs['cell_type'].isin(cell_types)].copy()
    sc.pp.filter_genes(adata, min_cells=4)
    sc.pp.filter_cells(adata, min_genes=100)
    cell_counts = adata.obs['cell_type'].value_counts()
    min_cells = cell_counts.min()
    print(f"Minimum cells per type: {min_cells}")
    print("Balancing classes...")
    balanced_adata = []
    for cell_type in cell_types:
        mask = adata.obs['cell_type'] == cell_type
        type_adata = adata[mask]
        if len(type_adata) > min_cells:
            balanced_adata.append(type_adata[np.random.choice(len(type_adata), min_cells, replace=False)])
        else:
            balanced_adata.append(type_adata)
    
    adata_balanced = balanced_adata[0].concatenate(balanced_adata[1:])
    
    output_h5ad = os.path.join(output_dir, 'processed.h5ad')
    adata_balanced.write(output_h5ad)

    if scipy.sparse.issparse(adata_balanced.X):
        data_array = adata_balanced.X.toarray()
    else:
        data_array = adata_balanced.X

    cell_types_series = pd.Series(adata_balanced.obs['cell_type'], name='cell_type')
    expression_df = pd.DataFrame(
        data_array,
        columns=adata_balanced.var_names,
        index=adata_balanced.obs_names
    )
    final_df = pd.concat([cell_types_series, expression_df], axis=1)

    output_csv = os.path.join(output_dir, 'processed_data.csv')
    final_df.to_csv(output_csv, index=False)
    



def make_datasets(input_file):
    print("Loading data...")
    adata_balanced = sc.read_h5ad(input_file)
    
    # Convert to dense array if sparse
    expressions = adata_balanced.X.toarray() if scipy.sparse.issparse(adata_balanced.X) else adata_balanced.X\
    
    cell_types = pd.Categorical(adata_balanced.obs['cell_type']).codes
    
    dataset = GeneExpressionDataset(expressions, cell_types)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    
    print(f"""
    Dataset splits:
    - Total cells: {total_size}
    - Training: {train_size}
    - Validation: {val_size}
    - Test: {test_size}
    """)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'n_genes': adata_balanced.n_vars,
        'n_cell_types': len(np.unique(cell_types))
    }

def create_dataloaders(datasets, batch_size=128):
    return {
        'train': DataLoader(datasets['train_dataset'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(datasets['val_dataset'], batch_size=batch_size),
        'test': DataLoader(datasets['test_dataset'], batch_size=batch_size)
    }

if __name__ == "__main__":
    preprocess_data('test.h5ad', '3class')
