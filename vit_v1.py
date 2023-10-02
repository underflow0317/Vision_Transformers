import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

#Vision Transformer模型
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
  
# 定义训练和微调函数
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

# 定义测试函数
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

# 设置超参数
num_classes = 10
patch_size = 8
hidden_dim = 960
num_heads = 12 #embed_dim 必须是 num_heads 的倍数
num_layers = 12
image_size = 32
batch_size = 64
num_epochs_pretrain = 30
num_epochs_finetune = 20
learning_rate_pretrain = 0.0001
learning_rate_finetune = 0.00001

#数据预处理和加载CIFAR-10数据集
#CIFAR-10:train=50000,test=10000
transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#train size:50000 / 64=781.25，大约782个batch
#使用loader確保數據分離
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test size:10000 / 64=156.25，大约157个batch
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型并定义损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(num_classes, patch_size, hidden_dim, num_heads, num_layers, image_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_pretrain = optim.Adam(model.parameters(), lr=learning_rate_pretrain, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=True)
'''
                    optim.RMSprop(
                        model.parameters(),
                        lr=learning_rate_pretrain,
                        alpha=0.99,           # 移动平均值的衰减率，根据经验通常设置为0.99
                        eps=1e-08,            # epsilon用于防止除零错误，通常使用默认值1e-08
                        weight_decay=0.001,   # 权重衰减（L2正则化），权重衰减（L2正则化）的参数
                        momentum=0,           # 动量是一个用于加速梯度下降的参数，通常在0到1之间，不使用动量设置为0
                        centered=False        # 中心化的计算梯度的移动平均值，并将其用于更新参数，不确定可以将它设置为False
                    )
'''
optimizer_finetune = optim.Adam(model.parameters(), lr=learning_rate_finetune, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=True)


######### Pre-train
for epoch in range(num_epochs_pretrain):
    train_loss = train_model(model, train_loader, optimizer_pretrain, criterion, device)
    print(f"Pretraining - Epoch {epoch + 1}/{num_epochs_pretrain}, Loss: {train_loss:.4f}")

# 保存预训练模型
torch.save(model.state_dict(), "vit_pretrained.pth")


######### Fine-tune
model.load_state_dict(torch.load("vit_pretrained.pth"))
for param in model.parameters():
    param.requires_grad = True

for epoch in range(num_epochs_finetune):
    train_loss = train_model(model, train_loader, optimizer_finetune, criterion, device)
    test_loss, accuracy = test_model(model, test_loader, criterion, device)
    print(f"Fine-tuning - Epoch {epoch + 1}/{num_epochs_finetune}, "
          f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

# 保存微调后的模型
torch.save(model.state_dict(), "vit_finetuned.pth")
