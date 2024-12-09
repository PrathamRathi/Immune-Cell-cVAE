import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import preprocess

NUM_CLASSES = 4
CLASS_LOSS_ALPHA = 1
KL_LOSS_ALPHA = 1

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = num_heads * head_size
        self.query = nn.Linear(input_size, self.hidden_size)
        self.key = nn.Linear(input_size, self.hidden_size)
        self.value = nn.Linear(input_size, self.hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
        )
        self.projection = nn.Linear(self.hidden_size, input_size)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_output, _ = self.attention(q,k,v)
        proj_att_output = self.projection(attention_output)
        return proj_att_output

class cVAE_Att(nn.Module):
    def __init__(self, input_dim, n_conditions, latent_dim=32, hidden_dims=[256,128], num_heads=8):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_dims = [input_dim + n_conditions] + hidden_dims
        encoder_layers = []
        
        for i in range(len(encoder_dims) - 1):
            encoder_layers.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i + 1]),
                nn.BatchNorm1d(encoder_dims[i + 1]),
                nn.ReLU(),
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Add attention layer after encoder
        head_size = int(128 / num_heads)
        self.attention = MultiHeadAttention(latent_dim, num_heads, head_size)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, n_conditions),
        )

        # Latent space
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (same as before)
        decoder_dims = [latent_dim + n_conditions] + list(reversed(hidden_dims)) + [input_dim]
        decoder_layers = []
        
        for i in range(len(decoder_dims) - 2):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                nn.BatchNorm1d(decoder_dims[i + 1]),
                nn.ReLU(),
            ])
        
        decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, c):
        # Concatenate input and condition
        x_c = torch.cat([x, c], dim=1)
        h = self.encoder(x_c)
        
        # Apply attention
        mu = self.mu(h)
        log_var = self.log_var(h)
        
        mu_attend = self.attention(mu)
        log_var_attend = self.attention(log_var)
        return mu_attend, log_var_attend

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
            # Concatenate latent and condition
            z_c = torch.cat([z, c], dim=1)
            return self.decoder(z_c)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        cell_type_pred = self.classifier(z)
        return self.decode(z, c), mu, log_var, cell_type_pred

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_recon_loss = 0
    total_kl_loss = 0
    total_class_loss = 0
    num_batches = 0
    
    for x, c in train_loader:
        x, c = x.to(device), c.to(device)
        
        # Convert cell types to one-hot
        c_onehot = F.one_hot(c, num_classes=NUM_CLASSES).float() 
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_x, mu, log_var, cell_type_pred = model(x, c_onehot)
        
        # Calculate losses
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        class_loss = F.cross_entropy(cell_type_pred, c_onehot, reduction='mean')

        loss = recon_loss + KL_LOSS_ALPHA * kl_loss + CLASS_LOSS_ALPHA * class_loss
        
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_class_loss += class_loss.item()
        num_batches += 1
    
    return total_recon_loss / num_batches, total_kl_loss / num_batches, total_class_loss / num_batches

def validate(model, val_loader, device):
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    total_class_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, c in val_loader:
            x, c = x.to(device), c.to(device)
            
            # Convert cell types to one-hot
            c_onehot = F.one_hot(c, num_classes=NUM_CLASSES).float() 
            
            # Forward pass
            recon_x, mu, log_var, cell_type_pred = model(x, c_onehot)
            
            # Calculate losses
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
            class_loss = F.cross_entropy(cell_type_pred, c_onehot,reduction='mean')

            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_class_loss += class_loss.item()
            num_batches += 1
    
    return total_recon_loss / num_batches, total_kl_loss / num_batches, total_class_loss / num_batches

def train_cvae(train_loader, val_loader, input_dim, n_conditions=4, epochs=100, name='attModel'):
    # Use MPS (Metal Performance Shaders) for M2 Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = cVAE_Att(input_dim=input_dim, n_conditions=n_conditions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_losses = []
    val_losses = []
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):        
        train_recon_loss, train_kl_loss, train_class_loss = train_epoch(model, train_loader, optimizer, device)
        
        val_recon_loss, val_kl_loss, val_class_loss = validate(model, val_loader, device)
        
        train_total_loss = train_recon_loss + train_kl_loss + train_class_loss
        train_losses.append(train_total_loss)
    
        val_total_loss = val_recon_loss + val_kl_loss + val_class_loss
        val_losses.append(val_total_loss)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train - Recon Loss: {train_recon_loss:.4f}, KL Loss: {train_kl_loss:.4f}, Class Loss: {train_class_loss:.4f}")
        print(f"Val   - Recon Loss: {val_recon_loss:.4f}, KL Loss: {val_kl_loss:.4f}, Class Loss: {val_class_loss:.4f}")
        
        interim_name = 'best_cvae_att_' + name + '.pt'
        # Save best model
        if val_total_loss < best_val_loss:
            'saving interim model'
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), interim_name)
    
    plot_loss_vs_epoch(train_losses, name + "train_loss")
    plot_loss_vs_epoch(val_losses, name + "val_loss", type='Validation')
    return model

def plot_loss_vs_epoch(losses, filename, type='Train'):
    # Ensure the filename ends with .png for saving as an image
    if not filename.endswith('.png'):
        filename += '.png'

    # Create the epochs array
    epochs = range(1, len(losses) + 1)
    
    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(type + ' Loss vs. Epoch')
    plt.grid(True)
    plt.legend()
    
    # Save the plot to the given filename
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    print('training balanced')
    data_path = 'processed_data/processed.h5ad'
    datasets = preprocess.make_datasets(data_path)
    dataloaders = preprocess.create_dataloaders(datasets, batch_size=64)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    print(sample_batch.shape)
    name = 'balanced'
    # Train model
    model = train_cvae(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        n_conditions=NUM_CLASSES,  # number of cell types
        epochs=30,
        name=name
    )
    print('saving final model')
    final_name = 'final_cvae_att_' + name + '.pt'
    torch.save(model.state_dict(), final_name)