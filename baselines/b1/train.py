import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from model import Baseline1
import torch.nn as nn
from data_utils.dataloader import get_data_loaders
from helper_utils.evaluation import  evaluate_test_set

def train_epoch(model,data_loader,criterion,optimizer,device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    loader = tqdm(data_loader,desc="")
    for image,label in loader:
        #To device
        image, label = image.to(device), label.to(device)

        #Model training
        output = model(image)
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #accumulating losses and correct predictions
        _, predicted = torch.max(output.data,1)
        correct += (predicted == label).sum().item()
        total += label.size(0)
        total_loss += loss.item()

        #tqdm utility
        loader.set_postfix(loss=loss.item(), acc=100. * correct / total)

    #accuracy and loss per epoch calculation
    acc_epoch = 100 * correct / total
    loss_epoch = total_loss / len(data_loader)

    return loss_epoch , acc_epoch

def eval_epoch(model,data_loader,criterion,device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in data_loader:
            # To device
            image, label = image.to(device), label.to(device)

            #Predictions
            output = model(image)
            loss = criterion(output, label)

            #accumilating losses and correct predictions
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()

        # accuracy and loss per epoch calculation
        acc_epoch = 100 * correct / total
        loss_epoch = total_loss / len(data_loader)

    return loss_epoch, acc_epoch


def train(cfg):
    # Prepare parameters
    save_path = "models/b1/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["model"]["num_classes"]
    dropout = cfg["training"]["dropout"]
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]
    epochs = cfg["training"]["epochs"]

    model = Baseline1(num_classes=num_classes,
                      dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Retrieve data
    train_loader, val_loader, test_loader = get_data_loaders(cfg, "spatial")

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} ----> Train loss: {train_loss} | Train acc: {train_acc}")
        print(f"Epoch {epoch + 1} ----> Val loss: {val_loss} | Val acc: {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model ({best_acc:.2f}%)")

    #Evaluate on test set
    model.load_state_dict(torch.load(save_path))
    evaluate_test_set(model, test_loader, device, cfg["model"]["class_names"])

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)