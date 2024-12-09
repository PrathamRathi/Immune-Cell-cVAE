import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess 
from tqdm import tqdm 

NUM_CLASSES = 4

class CellTypeClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=[128, 64]):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            
        layers.append(nn.Linear(dims[-1], n_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x)

def train_classifier(model, train_loader, val_loader, epochs=25, learning_rate=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, c in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, c = x.to(device), c.to(device)  
            c_onehot = F.one_hot(c, num_classes=NUM_CLASSES).float() 
            optimizer.zero_grad()
            outputs = model(x)
            loss = F.cross_entropy(outputs, c_onehot)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = torch.max(outputs, dim=1).indices
            correct += (pred == c).sum().item()
            total += c.size(0)
            total_loss += loss.item()
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_acc = evaluate(model, val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_classifier.pt')
    
    return model

def evaluate(model, data_loader):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, c in data_loader:
            x, c = x.to(device), c.to(device)
            outputs = model(x)
            pred = torch.max(outputs, dim=1).indices
            correct += (pred == c).sum().item()
            total += c.size(0)
    
    return correct / total

def test_classifier(model, test_loader):
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

if __name__ == "__main__":
    data_path = 'processed_data/processed.h5ad'
    datasets = preprocess.make_datasets(data_path)
    dataloaders = preprocess.create_dataloaders(datasets, batch_size=64)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    print(sample_batch.shape)

    
    # Initialize model
    model = CellTypeClassifier(input_dim=input_dim, n_classes=4)
    test_loader = dataloaders['test']
    # Train
    model = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5
    )
    
    # Test
    test_acc = test_classifier(model, test_loader)
