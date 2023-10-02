import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import torch.utils.checkpoint as checkpoint

# 数据加载和预处理
def load_data(batch_size, dataset_name, use_half_data=False):
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 选择数据集
    if dataset_name == "coco":
        coco_root = './data/coco'
        ann_file_train = './data/coco/annotations/instances_train2017.json'
        ann_file_val = './data/coco/annotations/instances_val2017.json'

        if use_half_data:
            # 使用COCO数据集的一半数据
            train_dataset = torchvision.datasets.CocoDetection(root=coco_root, annFile=ann_file_train, transform=transform_train)
            val_dataset = torchvision.datasets.CocoDetection(root=coco_root, annFile=ann_file_val, transform=transform_test)
            
            # 获取数据的前一半
            if len(train_dataset) % 2 == 0:
                half_len = len(train_dataset) // 2
            else:
                half_len = (len(train_dataset) // 2) + 1
            train_dataset, _ = torch.utils.data.random_split(train_dataset, [half_len, len(train_dataset) - half_len])

        else:
            train_dataset = torchvision.datasets.CocoDetection(root=coco_root, annFile=ann_file_train, transform=transform_train)
            val_dataset = torchvision.datasets.CocoDetection(root=coco_root, annFile=ann_file_val, transform=transform_test)

    elif dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=False)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=False)

    # 使用 DataLoader 加载数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=True, drop_last=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True, drop_last=True)

    return train_loader, test_loader

# 定义 Vision Transformer 模型
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
        # 使用 checkpoint.checkpoint 包装需要检查点的部分
        x = checkpoint.checkpoint(self._forward_impl, x)
        return x

    def _forward_impl(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1, x.size(1))  # 为了 Transformer 重塑张量形状 (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # 为了 Transformer 重塑张量形状 (seq_len, batch_size, hidden_dim)
        x = self.transformer(x, x)  # 传递 x 作为源和目标
        x = x.permute(1, 0, 2)  # 重塑回 (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.fc(x)
        return x

# 定义训练函数
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

def main():
    # 设置超参数
    num_classes = 10
    patch_size = 8
    hidden_dim = 120
    num_heads = 6  # embed_dim 必须是 num_heads 的倍数
    num_layers = 3
    image_size = 32
    batch_size = 2
    num_epochs_pretrain = 30
    num_epochs_finetune = 20
    learning_rate_pretrain = 0.0001
    learning_rate_finetune = 0.00001

    ##########Pre-train
    # 数据加载
    coco_train_loader, coco_test_loader = load_data(batch_size, dataset_name="coco")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建并加载 COCO 预训练模型
    pretrained_model = VisionTransformer(num_classes, patch_size, hidden_dim, num_heads, num_layers, image_size).to(device)

    # 定义优化器和损失函数
    optimizer_pretrain = optim.Adam(pretrained_model.parameters(), lr=learning_rate_pretrain, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0.001, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    # 预训练模型
    for epoch in range(num_epochs_pretrain):
        train_loss = train_model(pretrained_model, coco_train_loader, optimizer_pretrain, criterion, device)
        print(f"Pretraining - Epoch {epoch + 1}/{num_epochs_pretrain}, Loss: {train_loss:.4f}")
        torch.cuda.empty_cache()  # 手动释放GPU内存

    # 保存预训练模型
    torch.save(pretrained_model.state_dict(), "vit_pretrained.pth")

    ##########Fine-tune
    # 数据加载
    cifar10_train_loader, cifar10_test_loader = load_data(batch_size, dataset_name="cifar10")

    # 创建模型并加载预训练权重
    model = VisionTransformer(num_classes, patch_size, hidden_dim, num_heads, num_layers, image_size).to(device)
    model.load_state_dict(torch.load("vit_pretrained.pth"))

    # 定义微调优化器
    optimizer_finetune = optim.Adam(model.parameters(), lr=learning_rate_finetune, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0.001, amsgrad=True)

    # 微调模型
    for epoch in range(num_epochs_finetune):
        train_loss = train_model(model, cifar10_train_loader, optimizer_finetune, criterion, device)
        test_loss, accuracy = test_model(model, cifar10_test_loader, criterion, device)
        print(f"Fine-tuning - Epoch {epoch + 1}/{num_epochs_finetune}, "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    # 保存微调后的模型
    torch.save(model.state_dict(), "vit_finetuned.pth")

if __name__ == '__main__':
    main()  # entry function
