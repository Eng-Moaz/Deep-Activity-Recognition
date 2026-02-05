import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_test_set(model, data_loader, device, classes,save_path=None):
    model.eval()
    y_true = []
    y_pred = []

    print("Running detailed evaluation on Test Set...")

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Baseline 1 (Spatial)')
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion Matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return accuracy