import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from cVAE import cVAE
from cVAE_Att import cVAE_Att
import preprocess
from scipy import stats

os.environ['MKL_NUM_THREADS'] = '1'  # To avoid potential hardware conflicts
os.environ['OPENBLAS_NUM_THREADS'] = '1'

NUM_CLASSES = 4
CELL_TYPE_NAMES = ['Monocyte', 'CD4 T cell', 'CD8 T cell','NK cell']

def calculate_accuracy(predictions, true_labels):
    # Convert probabilities to predicted class indices
    predicted_classes = np.argmax(predictions, axis=1)
    correct = np.sum(predicted_classes == true_labels)
    total = true_labels.shape[0]
    accuracy = correct / total
    return accuracy

def evaluate_cvae(model, test_loader, device):
    """
    Evaluate cVAE and generate reconstructed data for all test samples.
    """
    model.eval()
    original_data = []
    reconstructed_data = []
    cell_types = []
    classifications = []

    with torch.no_grad():
        for x, c in test_loader:
            x, c = x.to(device), c.to(device)
            c_onehot = torch.nn.functional.one_hot(c, num_classes=NUM_CLASSES).float()
            
            # Get reconstruction
            recon_x, _, _ , class_pred = model(x, c_onehot)
            
            classifications.append(class_pred.cpu().numpy())
            original_data.append(x.cpu().numpy())
            reconstructed_data.append(recon_x.cpu().numpy())
            cell_types.append(c.cpu().numpy())
    
    # Concatenate all batches
    original_data = np.concatenate(original_data, axis=0)
    reconstructed_data = np.concatenate(reconstructed_data, axis=0)
    cell_types = np.concatenate(cell_types, axis=0)
    classifications = np.concatenate(classifications, axis=0)

    return original_data, reconstructed_data, cell_types, classifications

def plot_dimension_reduction(original_data, reconstructed_data, cell_types, save_path=None):
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    scaler = StandardScaler()
    original_scaled = original_data#scaler.fit_transform(original_data)
    reconstructed_scaled = reconstructed_data#scaler.transform(reconstructed_data)
    
    try:
        reducer = PCA(n_components=2, random_state=42)
        print("Using PCA for dimensionality reduction...")
        original_embedding = reducer.fit_transform(original_scaled)
        reconstructed_embedding = reducer.fit_transform(reconstructed_scaled)
        
        cell_type_names = CELL_TYPE_NAMES
        colors = sns.color_palette('husl', n_colors=len(cell_type_names))
        
        # Plot original data
        for i, cell_type in enumerate(range(len(cell_type_names))):
            mask = cell_types == cell_type
            ax1.scatter(original_embedding[mask, 0], original_embedding[mask, 1], 
                       c=[colors[i]], label=cell_type_names[i], alpha=0.6)
        ax1.set_title(f'Original Data PCA')
        ax1.legend()
        
        # Plot reconstructed data
        for i, cell_type in enumerate(range(len(cell_type_names))):
            mask = cell_types == cell_type
            ax2.scatter(reconstructed_embedding[mask, 0], reconstructed_embedding[mask, 1], 
                       c=[colors[i]], label=cell_type_names[i], alpha=0.6)
        ax2.set_title(f'Reconstructed Data PCA')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"Error during dimensionality reduction: {str(e)}")

def spearman_correlation(original, reconstructed):
    correlations = []
    for i in range(original.shape[0]):
        corr, _ = stats.spearmanr(original[i], 
                                 reconstructed[i])
        correlations.append(corr)
    return np.mean(correlations)

def compute_metrics(original_data, reconstructed_data, cell_types):
    """
    Compute evaluation metrics for the reconstructions.
    """
    # Mean squared error per cell type
    mse_by_type = {}
    for i, cell_type in enumerate(CELL_TYPE_NAMES):
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

    model = cVAE(input_dim=input_dim, n_conditions=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('best_cvae_model.pt'))
    model.eval()
    pca_path = "pca_cvae.png"

    # model = cVAE_Att(input_dim=input_dim, n_conditions=NUM_CLASSES).to(device)
    # model.load_state_dict(torch.load('best_cvae_att_balanced.pt'))
    # model.eval()
    # pca_path = "pca_bestbalanced.png"

    # Evaluate model
    print("Evaluating model...")
    original_data, reconstructed_data, labels, classifications = evaluate_cvae(model, test_loader, device)
    
    print("\nGenerating PCA visualization...")
    plot_dimension_reduction(original_data, reconstructed_data, labels, 
                          save_path=pca_path)
    
    print("\nComputing metrics...")
    compute_metrics(original_data, reconstructed_data, labels)

    accuracy = calculate_accuracy(classifications, labels)
    print("\nAccuracy: " + str(accuracy))

if __name__ == "__main__":
    main()

