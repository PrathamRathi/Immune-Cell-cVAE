import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from cVAE import cVAE
import preprocess
os.environ['MKL_NUM_THREADS'] = '1'  # To avoid potential hardware conflicts
os.environ['OPENBLAS_NUM_THREADS'] = '1'
NUM_CLASSES = 3

def evaluate_cvae(model, test_loader, device):
    """
    Evaluate cVAE and generate reconstructed data for all test samples.
    """
    model.eval()
    original_data = []
    reconstructed_data = []
    cell_types = []
    
    with torch.no_grad():
        for x, c in test_loader:
            x, c = x.to(device), c.to(device)
            c_onehot = torch.nn.functional.one_hot(c, num_classes=NUM_CLASSES).float()
            
            # Get reconstruction
            recon_x, _, _ , _= model(x, c_onehot)
            
            # Store original and reconstructed data
            original_data.append(x.cpu().numpy())
            reconstructed_data.append(recon_x.cpu().numpy())
            cell_types.append(c.cpu().numpy())
    
    # Concatenate all batches
    original_data = np.concatenate(original_data, axis=0)
    reconstructed_data = np.concatenate(reconstructed_data, axis=0)
    cell_types = np.concatenate(cell_types, axis=0)
    
    return original_data, reconstructed_data, cell_types

def plot_dimension_reduction(original_data, reconstructed_data, cell_types, method='pca', save_path=None):
    """
    Create PCA visualizations for original and reconstructed data.
    Optionally try UMAP if available and requested.
    """
    # Set up plot style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Scale data
    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original_data)
    reconstructed_scaled = scaler.transform(reconstructed_data)
    
    try:
        # if method.lower() == 'umap':
        #     # Try importing UMAP - if it fails, fall back to PCA
        #     try:
        #         print('trying to import UMAP')
        #         from umap import UMAP
        #         print('imported')
        #         reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        #         print("Using UMAP for dimensionality reduction...")
        #     except:
        #         print("UMAP import failed, falling back to PCA...")
        #         reducer = PCA(n_components=2, random_state=42)
        # else:
        reducer = PCA(n_components=2, random_state=42)
        print("Using PCA for dimensionality reduction...")
        
        # Compute embeddings
        print("Computing embedding for original data...")
        original_embedding = reducer.fit_transform(original_scaled)
        print("Computing embedding for reconstructed data...")
        reconstructed_embedding = reducer.fit_transform(reconstructed_scaled)
        
        # Color mapping for cell types
        cell_type_names = ['Monocyte', 'CD4 T cell', 'NK cell']
        colors = sns.color_palette('husl', n_colors=len(cell_type_names))
        
        # Plot original data
        for i, cell_type in enumerate(range(len(cell_type_names))):
            mask = cell_types == cell_type
            ax1.scatter(original_embedding[mask, 0], original_embedding[mask, 1], 
                       c=[colors[i]], label=cell_type_names[i], alpha=0.6)
        ax1.set_title(f'Original Data ({method.upper()})')
        ax1.legend()
        
        # Plot reconstructed data
        for i, cell_type in enumerate(range(len(cell_type_names))):
            mask = cell_types == cell_type
            ax2.scatter(reconstructed_embedding[mask, 0], reconstructed_embedding[mask, 1], 
                       c=[colors[i]], label=cell_type_names[i], alpha=0.6)
        ax2.set_title(f'Reconstructed Data ({method.upper()})')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"Error during dimensionality reduction: {str(e)}")
        print("Please try using method='pca' if you encountered issues with UMAP")

def compute_metrics(original_data, reconstructed_data, cell_types):
    """
    Compute evaluation metrics for the reconstructions.
    """
    # Mean squared error per cell type
    mse_by_type = {}
    for i, cell_type in enumerate(['Monocyte', 'CD4 T cell', 'NK cell']):
        mask = cell_types == i
        mse = np.mean((original_data[mask] - reconstructed_data[mask]) ** 2)
        mse_by_type[cell_type] = mse
    
    # Overall MSE
    total_mse = np.mean((original_data - reconstructed_data) ** 2)
    
    # Correlation between original and reconstructed data
    correlations = []
    for i in range(len(original_data)):
        try:
            corr = np.corrcoef(original_data[i], reconstructed_data[i])[0,1]
            if not np.isnan(corr):
                correlations.append(corr)
        except:
            continue
    
    mean_correlation = np.mean(correlations)
    
    print("\nReconstruction Metrics:")
    print(f"Overall MSE: {total_mse:.4f}")
    print(f"Mean Correlation: {mean_correlation:.4f}")
    print("\nMSE by Cell Type:")
    for cell_type, mse in mse_by_type.items():
        print(f"{cell_type}: {mse:.4f}")

def main():
    # Load trained model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = 365
    data_path = 'processed_data/processed.h5ad'
    datasets = preprocess.make_datasets(data_path)
    dataloaders = preprocess.create_dataloaders(datasets)

    test_loader = dataloaders['test']

    # Load your model (adjust parameters as needed)
    model = cVAE(input_dim=input_dim, n_conditions=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('best_cvae_model.pt'))
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    original_data, reconstructed_data, cell_types = evaluate_cvae(model, test_loader, device)
    
    # Try both PCA and UMAP
    print("\nGenerating PCA visualization...")
    plot_dimension_reduction(original_data, reconstructed_data, cell_types, 
                           method='pca', save_path='pca_comparison.png')
    
    # Compute and display metrics
    print("\nComputing metrics...")
    compute_metrics(original_data, reconstructed_data, cell_types)

if __name__ == "__main__":
    main()

