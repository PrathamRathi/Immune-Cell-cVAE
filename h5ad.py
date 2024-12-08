import scanpy as sc

# # Load the file
# adata = sc.read_h5ad('test.h5ad')

# # View the main contents
# print(adata.shape)  # (Number of cells, Number of genes)


# # Access specific attributes
# print("Observations (cells):", adata.obs)  # Metadata for cells
# print("Variables (genes):", adata.var)    # Metadata for genes
# print("Expression matrix:", adata.X)     # Gene expression data
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the h5ad file
filename = 'test.h5ad'  # Replace with the path to your file
adata = sc.read_h5ad(filename)

# Step 2: Check for raw data
if adata.raw is not None:
    print("Raw data is available.")
else:
    print("Raw data is not available.")
    exit()  # Exit if no raw data exists

# Step 3: Access the raw expression matrix
raw_matrix = adata.raw.X  # This might be sparse

# Step 4: Convert sparse matrix to dense (if necessary)
raw_dense = raw_matrix.toarray() if not isinstance(raw_matrix, np.ndarray) else raw_matrix

print("Total genes in raw data:", adata.raw.n_vars)  # Number of genes
print("Gene names (first 10):", adata.raw.var_names[:10])  # Sample of gene names

# Step 5: View raw gene metadata
print("Raw Gene Metadata:")
print(adata.raw.var.head())

# Step 6: View raw cell metadata
print("Raw Cell Metadata:")
print(adata.obs.head())

# Step 7: Inspect a subset of raw data
print("First 10 genes for the first 5 cells:")
print(raw_dense[:5, :10])

# Step 8: Convert to DataFrame for visualization
raw_df = pd.DataFrame(raw_dense, index=adata.obs_names, columns=adata.raw.var_names)

# Step 9: Visualize raw data
# Heatmap of expression for the first 10 cells and genes
sns.heatmap(raw_df.iloc[:10, :10], cmap="viridis")
plt.title("Heatmap of Gene Expression (Subset)")
plt.xlabel("Genes")
plt.ylabel("Cells")
plt.show()

# Distribution of expression levels for a specific gene
gene_to_plot = adata.raw.var_names[0]  # Change index for a different gene
plt.hist(raw_df[gene_to_plot], bins=50, color='skyblue', edgecolor='black')
plt.title(f"Distribution of Expression Levels for {gene_to_plot}")
plt.xlabel("Expression Level")
plt.ylabel("Frequency")
plt.show()

# Save the subset of raw data to a CSV file for external inspection
subset = raw_df.iloc[:100, :100]  # Adjust as needed
subset.to_csv("raw_data_subset.csv")
print("Subset of raw data saved to 'raw_data_subset.csv'.")


# Step 2: Check if 'cell_type' metadata exists
if 'cell_type' not in adata.obs:
    print("No 'cell_type' information available in the dataset.")
    exit()

# Step 3: Group by cell type and count the occurrences
cell_type_counts = adata.obs['cell_type'].value_counts()
print("Cell type counts:")
print(cell_type_counts)

# Step 4: Save cell type counts to a CSV file
cell_type_counts.to_csv("cell_type_counts.csv", header=['Count'])
print("Cell type counts saved to 'cell_type_counts.csv'.")

# Step 5 (Optional): Add the cell type as a new column in the raw data subset
raw_df = pd.DataFrame(
    adata.raw.X, 
    index=adata.obs_names, 
    columns=adata.raw.var_names
)

# Add cell type information
raw_df['cell_type'] = adata.obs['cell_type']

# Save the raw data with cell types to a CSV file
raw_df.to_csv("raw_data_with_cell_types.csv")
print("Raw data with cell types saved to 'raw_data_with_cell_types.csv'.")
