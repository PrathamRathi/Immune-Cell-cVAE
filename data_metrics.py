import scanpy as sc
import pandas as pd
import os
import scipy.sparse
import numpy as np
def process_h5ad_data(input_file, cell_types, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    adata = sc.read_h5ad(input_file)
    
    print(f"Initial dataset shape: {adata.shape}")

    cell_type_col = 'cell_type'
    
    adata = adata[adata.obs[cell_type_col].isin(cell_types)].copy()
    inspect_adata(adata)
    print(f"Shape after filtering: {adata.shape}")
    
    print("\nAvailable layers in filtered data:")
    if adata.layers:
        for layer_name in adata.layers.keys():
            print(f"- {layer_name}")
    else:
        print("No layers found in the dataset")
    
    print("\nExporting raw counts...")
    raw_df = pd.DataFrame(
        adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    raw_df['cell_type'] = adata.obs[cell_type_col]
    raw_df.to_csv(os.path.join(output_dir, 'raw_counts.csv'))
    
    if adata.layers:
        for layer_name, layer_data in adata.layers.items():
            print(f"Exporting {layer_name} layer...")
            layer_df = pd.DataFrame(
                layer_data.toarray() if scipy.sparse.issparse(layer_data) else layer_data,
                index=adata.obs_names,
                columns=adata.var_names
            )
            layer_df['cell_type'] = adata.obs[cell_type_col]
            layer_df.to_csv(os.path.join(output_dir, f'{layer_name}_counts.csv'))
def inspect_adata(adata):
    print("=" * 50)
    print("BASIC INFORMATION")
    print("=" * 50)
    print(f"Total number of cells: {adata.n_obs}")
    print(f"Total number of genes: {adata.n_vars}")
    print(f"Data type in main matrix: {type(adata.X).__name__}")
    if hasattr(adata.X, 'dtype'):
        print(f"Data dtype in main matrix: {adata.X.dtype}")
    
    print("\n" + "=" * 50)
    print("LAYERS")
    print("=" * 50)
    if adata.layers:
        for layer_name, layer in adata.layers.items():
            print(f"\nLayer: {layer_name}")
            print(f"- Type: {type(layer).__name__}")
            if hasattr(layer, 'dtype'):
                print(f"- dtype: {layer.dtype}")
            print(f"- Shape: {layer.shape}")
            try:
                if scipy.sparse.issparse(layer):
                    dense_sample = layer[:5].toarray()
                else:
                    dense_sample = layer[:5]
                print(f"- Non-zero elements: {np.count_nonzero(dense_sample) / dense_sample.size:.2%} (based on first 5 cells)")
            except:
                pass
    else:
        print("No layers found")
    
    print("\n" + "=" * 50)
    print("OBSERVATION (CELL) ANNOTATIONS")
    print("=" * 50)
    if adata.obs.columns.size > 0:
        for col in adata.obs.columns:
            n_unique = adata.obs[col].nunique()
            print(f"\nColumn: {col}")
            print(f"- Type: {adata.obs[col].dtype}")
            print(f"- Number of unique values: {n_unique}")
            if n_unique <= 10 and (adata.obs[col].dtype.name == 'category' or 
                                 adata.obs[col].dtype.name == 'object'):
                print("- Unique values:")
                for val in adata.obs[col].unique():
                    count = adata.obs[col].value_counts()[val]
                    print(f"  * {val}: {count} cells")
    else:
        print("No observation annotations found")
    
    print("\n" + "=" * 50)
    print("VARIABLE (GENE) ANNOTATIONS")
    print("=" * 50)
    if adata.var.columns.size > 0:
        for col in adata.var.columns:
            n_unique = adata.var[col].nunique()
            print(f"\nColumn: {col}")
            print(f"- Type: {adata.var[col].dtype}")
            print(f"- Number of unique values: {n_unique}")
            if n_unique <= 10 and (adata.var[col].dtype.name == 'category' or 
                                 adata.var[col].dtype.name == 'object'):
                print("- Unique values:")
                for val in adata.var[col].unique():
                    count = adata.var[col].value_counts()[val]
                    print(f"  * {val}: {count} genes")
    else:
        print("No variable annotations found")
    
    print("\n" + "=" * 50)
    print("UNS (UNSTRUCTURED DATA)")
    print("=" * 50)
    if adata.uns:
        for key in adata.uns.keys():
            print(f"\nKey: {key}")
            value = adata.uns[key]
            print(f"- Type: {type(value).__name__}")
            if isinstance(value, np.ndarray):
                print(f"- Shape: {value.shape}")
    else:
        print("No unstructured annotations found")
    
    print("\n" + "=" * 50)
    print("OBSM (OBSERVATION MATRICES)")
    print("=" * 50)
    if adata.obsm:
        for key in adata.obsm.keys():
            value = adata.obsm[key]
            print(f"\nKey: {key}")
            print(f"- Type: {type(value).__name__}")
            print(f"- Shape: {value.shape}")
    else:
        print("No observation matrices found")
    
    print("\n" + "=" * 50)
    print("VARM (VARIABLE MATRICES)")
    print("=" * 50)
    if adata.varm:
        for key in adata.varm.keys():
            value = adata.varm[key]
            print(f"\nKey: {key}")
            print(f"- Type: {type(value).__name__}")
            print(f"- Shape: {value.shape}")
    else:
        print("No variable matrices found")

if __name__ == "__main__":
    # Specify cell types to filter
    cell_types_to_keep = ['CD4 T cell'] #['Monocyte', 'CD4 T cell', 'NK cell']

    process_h5ad_data(
        input_file='test.h5ad',
        cell_types=cell_types_to_keep,
        output_dir='output'
    )