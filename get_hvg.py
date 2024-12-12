import pandas as pd
import scanpy as sc
import numpy as np

def process_gene_expression(csv_file):
    df = pd.read_csv(csv_file)

    cell_types = df['cell_type']
    gene_expression = df.drop('cell_type', axis=1)

    adata = sc.AnnData(gene_expression)

    adata.obs['cell_type'] = cell_types
    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    hvg = adata.var_names[adata.var['highly_variable']].tolist()

    return hvg


def common_hvg(hvg_test, hvg_syn):
    matching = list(set(hvg_test) & set(hvg_syn)) 
    print(matching)
    print(len(matching)/len(hvg_test))

if __name__ == "__main__":
    hvg_test = process_gene_expression('test_data.csv')
    names = ['final_cvae_att_KL0.7_class1.4_recon1_epochs15NoNormal', 'final_cvae_att_KL1_class0_recon0.7_epochs15NoNormal', 
        'final_cvae_att_KL1_class1_recon1_epochs15NoNormal']
    
    for name in names:
        name = name + '.csv'
        hvg_syn = process_gene_expression(name)
        common_hvg(hvg_test, hvg_syn)