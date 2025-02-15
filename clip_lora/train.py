import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, dataloader, lr=1e-4, epochs=10):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, texts in dataloader:
            logits_per_image, logits_per_text = model(images, texts)
            labels = torch.arange(len(images), device=images.device)

            loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts in dataloader:
            logits_per_image, logits_per_text = model(images, texts)
            preds = logits_per_image.argmax(dim=1)
            labels = torch.arange(len(images), device=images.device)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct / total:.2%}")
