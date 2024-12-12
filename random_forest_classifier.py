import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import preprocess

def convert_dataloader_to_numpy(dataloader):
    """Convert PyTorch DataLoader to numpy arrays"""
    all_features = []
    all_labels = []
    
    for features, labels in tqdm(dataloader, desc="Converting data"):
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        
        all_features.append(features)
        all_labels.append(labels)
    
    return np.vstack(all_features), np.concatenate(all_labels)

class RandomForestAnalyzer:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1 
        )
        
    def train(self, train_loader, val_loader=None):
        X_train, y_train = convert_dataloader_to_numpy(train_loader)
        self.model.fit(X_train, y_train)
        if val_loader is not None:
            X_val, y_val = convert_dataloader_to_numpy(val_loader)
            val_acc = self.model.score(X_val, y_val)
            print(f"Validation Accuracy: {val_acc:.4f}")
    
    def test(self, test_loader):
        X_test, y_test = convert_dataloader_to_numpy(test_loader)
        test_acc = self.model.score(X_test, y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc
    
    def plot_feature_importance(self, gene_names=None, top_n=20):
        importances = self.model.feature_importances_
        if gene_names is None:
            gene_names = [f"Gene_{i}" for i in range(len(importances))]

        gene_symbols = preprocess.get_gene_symbols(gene_names)
        importance_df = pd.DataFrame({
            'Gene': gene_names,
            'Importance': importances,
            'Gene Symbols': gene_symbols
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(top_n),
                   x='Importance', y='Gene Symbols')
        plt.title('Top Important Genes in Random Forest Classification')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Print top genes
        print("\nTop 20 Most Important Genes:")
        for _, row in importance_df.head(20).iterrows():
            print(f"{row['Gene Symbols']}: {row['Importance']:.4f}")
        
        return importance_df


def compare_hvg_rf_importance(adata, rf_importance_df, n_top=20):
    """Compare highly variable genes with Random Forest importance"""
    sc.pp.highly_variable_genes(adata, n_top_genes=len(adata.var_names))
    hvg_df = pd.DataFrame({
        'Gene': adata.var_names,
        'Highly_Variable': adata.var['highly_variable'],
        'HVG_Score': adata.var['dispersions_norm']
    })
    comparison_df = hvg_df.merge(
        rf_importance_df,
        on='Gene',
        how='inner'
    )
    correlation = np.corrcoef(
        comparison_df['HVG_Score'],
        comparison_df['Importance']
    )[0,1]
    
    top_hvg = comparison_df.nlargest(n_top, 'HVG_Score')
    top_rf = comparison_df.nlargest(n_top, 'Importance')
    hvg_set = set(preprocess.get_gene_symbols(top_hvg['Gene']))
    rf_set = set(preprocess.get_gene_symbols(top_rf['Gene']))
    overlap = hvg_set.intersection(rf_set)
    
    print('Correlation: ', correlation)
    print(f"\nOverlap Analysis of Top {n_top} Genes:")
    print(f"Number of overlapping genes: {len(overlap)}")
    print("\nGenes in both sets:")
    for gene in overlap:
        print(gene)
        
    print("\nTop HVG-only genes:")
    for gene in hvg_set - rf_set:
        print(gene)
        
    print("\nTop RF-only genes:")
    for gene in rf_set - hvg_set:
        print(gene)
    
    return comparison_df


if __name__ == "__main__":
    data_path = 'processed_data/processed.h5ad'
    datasets = preprocess.make_datasets(data_path)
    dataloaders = preprocess.create_dataloaders(datasets, batch_size=64)

    gene_names = datasets['gene_names']  
    

    rf_analyzer = RandomForestAnalyzer(n_estimators=100)
    rf_analyzer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val']
    )
    test_acc = rf_analyzer.test(dataloaders['test'])
    importance_df = rf_analyzer.plot_feature_importance(gene_names=gene_names)
    importance_df.to_csv('gene_importance_rankings.csv', index=False)
    # adata = sc.read_h5ad('processed_data/processed.h5ad')
    # comparison_df = compare_hvg_rf_importance(
    #     adata=adata,
    #     rf_importance_df=importance_df,
    #     n_top=20
    # )