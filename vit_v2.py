import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models.vision_transformer as vision_transformer
import matplotlib.pyplot as plt

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 110

    # Load CIFAR-10 dataset and create data loaders
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = 10  # CIFAR-10 has 10 classes

    print("Model: vit_b_16", flush=True)
    # Load the Vision Transformer model
    model = vision_transformer.vit_b_16(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Moving vit_b_16 to device", flush=True)
    model.to(device)

    train_acc_history = []
    test_acc_history = []

    print("Training epochs", flush=True)

    for epoch in range(7):  # You can adjust the number of training epochs as needed
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 110 == 109:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 110:.3f}", flush=True)
                running_loss = 0.0

        train_accuracy = 100 * correct_train / total_train
        train_acc_history.append(train_accuracy)

        # Testing the model's accuracy
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        test_acc_history.append(test_accuracy)

        print(f"Epoch {epoch + 1}: Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%", flush=True)

    print("Finished Fine-tuning")

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_vit_b_16.pth')

    # Plot training and test accuracy curves
    plt.plot(range(1, 8), train_acc_history, label='Train Accuracy')
    plt.plot(range(1, 8), test_acc_history, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
