import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Vision Transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes, patch_size, hidden_dim, num_heads, num_layers, image_size):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.embedding = nn.Sequential(
            nn.Conv2d(3, hidden_dim, patch_size, patch_size),
            nn.Flatten(),
            nn.Linear(hidden_dim * ((image_size // patch_size) ** 2), hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1, x.size(1))  # Reshape for transformer (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # Reshape for transformer (seq_len, batch_size, hidden_dim)
        x = self.transformer(x, x)  # Pass x as source and target
        x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Define training and fine-tuning functions
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return test_loss / len(test_loader), correct / total

def main():
    # Set hyperparameters
    num_classes = 10
    patch_size = 16  # Patch size (P) = 16
    hidden_dim = 768  # Latent vector (D). ViT-Base uses 768
    num_heads = 12  # embed_dim must be a multiple of num_heads. ViT-Base uses 12 heads
    num_layers = 12  # ViT-Base uses 12 encoder layer
    image_size = 224  #ori=32
    batch_size = 4
    num_epochs_pretrain = 30
    num_epochs_finetune = 20
    learning_rate_pretrain = 10e-3 # Base LR
    learning_rate_finetune = 10e-3

    # Data preprocessing and loading CIFAR-10 dataset
    # CIFAR-10: train=50000, test=10000
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train size: 50000 / 64 ≈ 782 batches
    # Use DataLoader to ensure data separation
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test size: 10000 / 64 ≈ 157 batches
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create the model and define loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(num_classes, patch_size, hidden_dim, num_heads, num_layers, image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_pretrain = optim.Adam(model.parameters(), lr=learning_rate_pretrain, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.03, amsgrad=True) # Weight decay for ViT-Base (on ImageNet-21k)
    #optimizer_finetune = optim.Adam(model.parameters(), lr=learning_rate_finetune, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.03, amsgrad=True)

    #### Pre-train
    for epoch in range(num_epochs_pretrain):
        train_loss = train_model(model, train_loader, optimizer_pretrain, criterion, device)
        print(f"Pretraining - Epoch {epoch + 1}/{num_epochs_pretrain}, Loss: {train_loss:.4f}")

    # Save the pre-trained model
    torch.save(model.state_dict(), "vit_pretrained.pth")

    '''
    #### Fine-tune
    model.load_state_dict(torch.load("vit_pretrained.pth"))
    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(num_epochs_finetune):
        train_loss = train_model(model, train_loader, optimizer_finetune, criterion, device)
        test_loss, accuracy = test_model(model, test_loader, criterion, device)
        print(f"Fine-tuning - Epoch {epoch + 1}/{num_epochs_finetune}, "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "vit_finetuned.pth")
    '''

if __name__ == '__main__':
    main()
