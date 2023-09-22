import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义Vision Transformer模型
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers):
        super(ViT, self).__init__()
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.view(x.size(0), -1, x.size(1))
        x = x + self.position_embedding
        
        # 使用自注意力层
        x = x.permute(1, 0, 2)  # 将batch维移到第2维
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # 恢复原始维度顺序
        
        x = x.mean(dim=1)  # Average pooling over patches
        x = self.fc(x)
        return x

# 超参数设置
image_size = 32
patch_size = 8
num_classes = 10
hidden_dim = 256
num_heads = 8
num_layers = 4
batch_size = 64
learning_rate = 0.001
epochs = 30

# 数据转换和加载
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100.0 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

# 在测试集上评估模型
model.eval()
total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_acc = 100.0 * correct / total
print(f"Test Loss: {total_loss / len(test_loader):.4f}, Test Acc: {test_acc:.2f}%")
