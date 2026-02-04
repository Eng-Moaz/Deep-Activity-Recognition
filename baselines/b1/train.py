import torch
import torch.optim as optim
from tqdm import tqdm
from model import Baseline1
import torch.nn as nn

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
        #accumilating losses and correct predictions
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
    #Prepare parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline1(num_classes=cfg["model"]["num_classes"],
                      dropout=cfg["training"]["dropout"])
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        train_loss , train_acc = train_epoch(model,train_loader,criterion,optimizer,device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} ----> Train loss: {train_loss} | Train acc: {train_acc}")
        print(f"Epoch {epoch+1} ----> Val loss: {val_loss} | Val acc: {val_acc}")

