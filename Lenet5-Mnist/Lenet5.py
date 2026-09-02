"""
LeNet-5 para reconhecimento de dígitos MNIST
=============================================
Implementação baseada no artigo:
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
"Gradient-Based Learning Applied to Document Recognition."
Proceedings of the IEEE, 86(11), 2278-2324.

Arquitetura original (Seção II.B do artigo):
  INPUT   32x32
  C1: conv 5x5      -> 6 mapas de 28x28
  S2: subsampling   -> 6 mapas de 14x14
  C3: conv 5x5      -> 16 mapas de 10x10
  S4: subsampling   -> 16 mapas de 5x5
  C5: conv 5x5      -> 120 (equivale a totalmente conectada, pois S4 já é 5x5)
  F6: totalmente conectada -> 84 unidades
  OUTPUT: 10 classes

Adaptações modernas feitas aqui (comuns em toda reimplementação atual):
  - Ativação ReLU no lugar da tanh escalada original (converge mais rápido).
  - Average pooling clássico mantido nas camadas S2/S4 (fiel ao artigo).
  - Camada de saída com Softmax/CrossEntropy no lugar das unidades RBF
    originais (as RBFs eram específicas do problema de 1998 e hoje são
    substituídas por uma camada densa + softmax em praticamente toda
    reimplementação, inclusive as oficiais do PyTorch/Keras).

Requisitos:
    pip install torch torchvision
"""

import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# 1. Definição da arquitetura LeNet-5
# ---------------------------------------------------------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # C1: entrada 1x32x32 -> 6x28x28  (kernel 5x5, sem padding)
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # S2: subamostragem 2x2 -> 6x14x14
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)                                                                                                                                                                                                               

        # C3: 6x14x14 -> 16x10x10
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # S4: subamostragem 2x2 -> 16x5x5
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: 16x5x5 -> 120x1x1 (equivale a full connection, pois o kernel
        # cobre exatamente o mapa inteiro)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # F6: totalmente conectada, 120 -> 84
        self.f6 = nn.Linear(120, 84)

        # Camada de saída: 84 -> num_classes
        self.output = nn.Linear(84, num_classes)

    def forward(self, x):
        # x esperado: (batch, 1, 32, 32)
        x = F.relu(self.c1(x))
        x = self.s2(x)

        x = F.relu(self.c3(x))
        x = self.s4(x)

        x = F.relu(self.c5(x))     # (batch, 120, 1, 1)
        x = torch.flatten(x, 1)    # (batch, 120)

        x = F.relu(self.f6(x))     # (batch, 84)
        x = self.output(x)         # (batch, num_classes) -- logits

        return x


def _read_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


class MNISTIdxDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, transform=None):
        self.images = _read_idx_images(images_path)
        self.labels = _read_idx_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]              # (28, 28) uint8
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)     # PIL/tensor conforme transform

        return image, label


def get_dataloaders(batch_size: int = 64, data_dir: str = "./data"):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(2),          # 28x28 -> 32x32
        transforms.ToTensor(),      # valores em [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)), 
    ])

    train_set = MNISTIdxDataset(
        images_path=os.path.join(data_dir, "train-images.idx3-ubyte"),
        labels_path=os.path.join(data_dir, "train-labels.idx1-ubyte"),
        transform=transform,
    )
    test_set = MNISTIdxDataset(
        images_path=os.path.join(data_dir, "t10k-images.idx3-ubyte"),
        labels_path=os.path.join(data_dir, "t10k-labels.idx1-ubyte"),
        transform=transform,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 3. Treinamento e avaliação
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    epochs = 10
    batch_size = 64
    learning_rate = 0.001

    PATH = 'C:/Users/Usuario/OneDrive/Documentos/GitHub/Ciencia de Dados/Lenet5-Mnist/data'

    # Ajuste "./data" para o caminho real da sua pasta, se necessário
    train_loader, test_loader = get_dataloaders(batch_size=batch_size,
                                                 data_dir=PATH)

    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Época {epoch:2d}/{epochs} | "
            f"Treino: loss={train_loss:.4f} acc={train_acc:.2f}% | "
            f"Teste: loss={test_loss:.4f} acc={test_acc:.2f}%"
        )

    # Salva os pesos treinados
    torch.save(model.state_dict(), "lenet5_mnist.pth")
    print("\nModelo treinado e salvo em 'lenet5_mnist.pth'.")


if __name__ == "__main__":
    main()