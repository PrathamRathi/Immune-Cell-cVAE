import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from cVAE_Att import cVAE_Att
import preprocess
from scipy import stats
from sklearn.manifold import TSNE
import pandas as pd

NUM_CLASSES = 4
CELL_TYPE_NAMES = ['Monocyte', 'CD4 T cell', 'CD8 T cell','NK cell']

def get_vectors(model, test_loader):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    
    original_data = []
    latent_vectors = []
    reconstructions = []
    true_labels = []
    
    with torch.no_grad():
        for x, c in test_loader:

            x = x.to(device)
            c = c.to(device)
            c_onehot = torch.nn.functional.one_hot(c, num_classes=NUM_CLASSES).float()
            z = model.get_latent_vector(x, c_onehot)
            recon = model.decode(z, c_onehot)
            
            latent_vectors.append(z.cpu().numpy())
            reconstructions.append(recon.cpu().numpy())
            true_labels.append(c.cpu().numpy())
            original_data.append(x.cpu().numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    reconstructions = np.vstack(reconstructions)
    true_labels = np.concatenate(true_labels)
    original_data = np.concatenate(original_data)
    
    return latent_vectors, reconstructions, true_labels, original_data

def PCA_analysis(latent_vectors, reconstructions, true_labels, original_data, n_components=2):
    latent_pca = PCA(n_components=n_components)
    latent_transformed = latent_pca.fit_transform(latent_vectors)
    
    recon_pca = PCA(n_components=n_components)
    recon_transformed = recon_pca.fit_transform(reconstructions)
    
    orig_pca = PCA(n_components=n_components)
    orig_transformed = orig_pca.fit_transform(original_data)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(15, 6))
    
    unique_labels = np.unique(true_labels)
    colors = sns.color_palette("husl", n_colors=len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        label_name = CELL_TYPE_NAMES[label]
        mask = true_labels == label
        ax1.scatter(latent_transformed[mask, 0], latent_transformed[mask, 1], 
                   c=[colors[i]], label=f'Cell Type {label_name}', alpha=0.7)
    ax1.set_title('PCA of Latent Space')
    ax1.set_xlabel(f'PC1 ({latent_pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({latent_pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.legend()
    
    for i, label in enumerate(unique_labels):
        label_name = CELL_TYPE_NAMES[label]
        mask = true_labels == label
        ax2.scatter(recon_transformed[mask, 0], recon_transformed[mask, 1],
                   c=[colors[i]], label=f'Cell Type {label_name}', alpha=0.7)
    ax2.set_title('PCA of Reconstructions')
    ax2.set_xlabel(f'PC1 ({recon_pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({recon_pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.legend()

    for i, label in enumerate(unique_labels):
        label_name = CELL_TYPE_NAMES[label]
        mask = true_labels == label
        ax3.scatter(orig_transformed[mask, 0], orig_transformed[mask, 1],
                   c=[colors[i]], label=f'Cell Type {label_name}', alpha=0.7)
    ax3.set_title('PCA of Original')
    ax3.set_xlabel(f'PC1 ({orig_pca.explained_variance_ratio_[0]:.2%} variance)')
    ax3.set_ylabel(f'PC2 ({orig_pca.explained_variance_ratio_[1]:.2%} variance)')
    ax3.legend()
    
    plt.tight_layout()
    
    total_latent_var = np.sum(latent_pca.explained_variance_ratio_)
    total_recon_var = np.sum(recon_pca.explained_variance_ratio_)
    total_orig_var = np.sum(orig_pca.explained_variance_ratio_)
    plt.show()
    results = {
        'latent_pca': latent_pca,
        'recon_pca': recon_pca,
        'latent_transformed': latent_transformed,
        'recon_transformed': recon_transformed,
        'orig_transformed': orig_transformed,
        'total_latent_variance_explained': total_latent_var,
        'total_recon_variance_explained': total_recon_var,
        'total_orig_variance_explained': total_orig_var,
        'true_labels': true_labels,
        'figure': fig
    }
    
    return results

def tsna_analysis(latent_vectors, reconstructions, true_labels, original_data, n_components=2, save_path=''):
    latent_TSNE = TSNE(n_components=n_components, perplexity=40, n_iter=300)
    latent_transformed = latent_TSNE.fit_transform(latent_vectors)
    
    recon_TSNE = TSNE(n_components=n_components, perplexity=40, n_iter=300)
    recon_transformed = recon_TSNE.fit_transform(reconstructions)
    
    orig_TSNE = TSNE(n_components=n_components, perplexity=40, n_iter=300)
    orig_transformed = orig_TSNE.fit_transform(original_data)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(15, 6))
    
    unique_labels = np.unique(true_labels)
    colors = sns.color_palette("husl", n_colors=len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        label_name = CELL_TYPE_NAMES[label]
        mask = true_labels == label
        ax1.scatter(latent_transformed[mask, 0], latent_transformed[mask, 1], 
                   c=[colors[i]], label=f'Cell Type {label_name}', alpha=0.7)
    ax1.set_title('TSNE of Latent Space')
    ax1.set_xlabel('TSNE 1')
    ax1.set_ylabel('TSNE 2')
    ax1.legend()
    
    for i, label in enumerate(unique_labels):
        label_name = CELL_TYPE_NAMES[label]
        mask = true_labels == label
        ax2.scatter(recon_transformed[mask, 0], recon_transformed[mask, 1],
                   c=[colors[i]], label=f'Cell Type {label_name}', alpha=0.7)
    ax2.set_title('TSNE of Reconstructions')
    ax1.set_xlabel('TSNE 1')
    ax1.set_ylabel('TSNE 2')
    ax2.legend()

    for i, label in enumerate(unique_labels):
        label_name = CELL_TYPE_NAMES[label]
        mask = true_labels == label
        ax3.scatter(orig_transformed[mask, 0], orig_transformed[mask, 1],
                   c=[colors[i]], label=f'Cell Type {label_name}', alpha=0.7)
    ax3.set_title('TSNE of Original')
    ax1.set_xlabel('TSNE 1')
    ax1.set_ylabel('TSNE 2')
    ax3.legend()
    
    plt.savefig(save_path)
    # plt.tight_layout()
    # plt.show()


def generate_samples(model, num_samples, num_classes, latent_dim, gene_symbols=[], output_file='generated_samples.csv'):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    
    all_samples = []
    all_labels = []
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            z = torch.randn(num_samples, latent_dim).to(device)
            
            labels = torch.nn.functional.one_hot(torch.full((num_samples,), class_idx, dtype=torch.long), num_classes=4).float().to(device)
            
            samples = model.decode(z, labels)
            
            samples_np = samples.cpu().numpy()
            all_samples.append(samples_np)
            all_labels.extend([class_idx] * num_samples)
    

    generated_samples = np.concatenate(all_samples, axis=0)
    labels = np.array(all_labels)
    cell_names = [CELL_TYPE_NAMES[i] for i in labels]

    sample_pca = PCA(n_components=2)
    sample_transformed = sample_pca.fit_transform(generated_samples)

    fig, (ax1) = plt.subplots(1, 1,  figsize=(10, 10))
    
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", n_colors=len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        label_name = CELL_TYPE_NAMES[label]
        mask = labels == label
        ax1.scatter(sample_transformed[mask, 0], sample_transformed[mask, 1], 
                   c=[colors[i]], label=f'Cell Type {label_name}', alpha=0.7)
        
    ax1.set_title('PCA of Generated Samples')
    ax1.set_xlabel(f'PC1')
    ax1.set_ylabel(f'PC2')
    ax1.legend()
    plt.show()

    sample_cols = gene_symbols
    df = pd.DataFrame(generated_samples, columns=sample_cols)
    df.insert(0, 'cell_type', cell_names)
    df = df.sort_values('cell_type')
    df.to_csv(output_file, index=False)
    
    return generated_samples, labels

def write_test_data(test_loader, gene_symbols, output_file='test_data.csv'):
    all_samples = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:   
            samples = data.cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_samples.append(samples)
            all_labels.append(labels)
    
    samples_array = np.concatenate(all_samples, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    
    sample_cols = gene_symbols
    cell_names = [CELL_TYPE_NAMES[i] for i in labels_array]
    
    df = pd.DataFrame(samples_array, columns=sample_cols)
    df.insert(0, 'cell_type', cell_names)
    df = df.sort_values('cell_type')
    df.to_csv(output_file, index=False)
    
    return samples_array, labels_array

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

input_dim = 365
data_path = 'processed_data/processed.h5ad'
datasets = preprocess.make_datasets(data_path)
dataloaders = preprocess.create_dataloaders(datasets)

test_loader = dataloaders['test']
gene_symbols = datasets['gene_symbols']

# model = cVAE(input_dim=input_dim, n_conditions=NUM_CLASSES).to(device)
# model.load_state_dict(torch.load('best_cvae_model.pt'))
# model.eval()

model = cVAE_Att(input_dim=input_dim, n_conditions=NUM_CLASSES).to(device)
print(model)
names = ['final_cvae_att_KL0.7_class1.4_recon1_epochs15NoNormal', 'final_cvae_att_KL1_class0_recon0.7_epochs15NoNormal', 
         'final_cvae_att_KL1_class1_recon1_epochs15NoNormal']

# for name in names:
#     base_name = name
#     model.load_state_dict(torch.load(base_name + '.pt'))

#     tsna_path = 'tsna/' + base_name + '.png'
#     latent_vectors, reconstructions, true_labels, original_data = get_vectors(model, test_loader)
#     # tsna_analysis(latent_vectors, reconstructions, true_labels, original_data, n_components=2, save_path=tsna_path)

#     # write_test_data(test_loader, gene_symbols)

#     samples, labels = generate_samples(
#         model=model,
#         num_samples=1100,  
#         num_classes=4,  
#         latent_dim=32,    
#         gene_symbols=gene_symbols,
#         output_file=base_name + '.csv'
#     )
