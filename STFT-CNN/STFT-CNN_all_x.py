import os
import time

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from STFT_CNN import MainModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_algorithm_snr_h5s(root_folder, mod_types):
    """
    Loads .h5 spectrogram files from a specific algorithm's snr_X folder,
    filtered by modulation type (FM, PM, HYBRID).

    Parameters:
    - root_folder (str): Path to the snr_X directory (e.g., .../preprocessed_images/cdae/snr_0)
    - mod_types (list): List of modulation categories to include, e.g., ['FM', 'PM']

    Returns:
    - X: np.ndarray of images
    - y: np.ndarray of labels (modulation names as strings)
    """
    X = []
    y = []

    for mod_type in mod_types:
        mod_path = os.path.join(root_folder, mod_type)
        if not os.path.exists(mod_path):
            print(f"⚠️ Warning: {mod_path} does not exist. Skipping.")
            continue

        print(f"📂 Loading from {mod_type}...")
        files = [f for f in os.listdir(mod_path) if f.endswith(".h5")]

        for file in tqdm(files, desc=f"   {mod_type}", unit="file"):
            mod_name = file[:-3]  # Strip '.h5'
            file_path = os.path.join(mod_path, file)

            try:
                with h5py.File(file_path, "r") as h5f:
                    if mod_name not in h5f:
                        print(
                            f"⚠️ Warning: No top-level group named '{mod_name}' in {file_path}"
                        )
                        continue
                    group = h5f[mod_name]
                    for key in group.keys():
                        img = np.array(group[key])
                        X.append(img)
                        y.append(mod_name)
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")

    return np.array(X), np.array(y)


def prepare_dataloader(X, y, batch_size=32, shuffle=False, num_workers=2, device="cpu"):
    # Convert NumPy arrays to PyTorch tensors
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    elif not isinstance(X, torch.Tensor):
        raise TypeError("Input X must be a NumPy array or PyTorch tensor")

    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long)
    elif not isinstance(y, torch.Tensor):
        raise TypeError("Labels y must be a NumPy array or PyTorch tensor")

    # Ensure X has four dimensions (N, C, H, W)
    if X.ndim == 3:  # If (N, H, W), add a channel dimension
        X = X.unsqueeze(1)  # (N, 1, H, W)
    elif X.ndim == 4 and X.shape[-1] in [1, 3]:  # (N, H, W, C) case
        X = X.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)

    # Move data to the correct device
    X, y = X.to(device), y.to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    return loader


def train_model(
    model,
    train_loader,
    device,
    criterion,
    optimizer,
    scheduler=None,  # 🔧 Optional scheduler added
    epochs=10,
    patience=3,
    min_delta=0.0,
):

    model.to(device)
    model.train()

    loss_history = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0.0

        # Progress bar for each epoch
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass: Ignore output_image, focus only on output_class
            output_class = model(inputs)

            # Classification loss
            loss = criterion(output_class, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Live loss display
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # 🔄 Scheduler step
        if scheduler:
            scheduler.step()

        # Early stopping
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break  # Stop training early

    return loss_history


def plot_loss_curve(loss_history, output_path, title="Training Loss Over Epochs"):
    epochs = len(loss_history)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker="o", label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(output_path + f"loss_curve.png")
    plt.clf()


def display_confusion_matrix(
    model, data_loader, device, output_path, class_names=None, title="Confusion Matrix"
):
    """
    Generate and display a normalized confusion matrix for a trained model.

    Parameters:
        model (torch.nn.Module): Trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation dataset.
        device (torch.device): Device to run evaluation on (CPU/GPU).
        class_names (list, optional): List of class names. If None, uses numeric indices.
        title (str): Title of the confusion matrix plot.
    """
    # Switch model to evaluation mode
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # Disable gradient calculations for inference
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass: Ignore output_image, focus only on output_class
            output_class = model(inputs)

            # Get predicted class labels
            preds = torch.argmax(output_class, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = cm.shape[0]

    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100

    # If class_names isn't provided, use numeric class indices
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Plotting the confusion matrix
    plt.figure(
        figsize=(max(10, num_classes * 0.8), max(8, num_classes * 0.6))
    )  # Dynamic size
    im = plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=14)
    plt.colorbar(im, label="Percentage")  # Add colorbar with label

    # Create tick marks for class labels
    tick_marks = np.arange(num_classes)
    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right",
        va="top",
        fontsize=max(8, 12 - num_classes // 5),
    )
    plt.yticks(tick_marks, class_names, fontsize=max(8, 12 - num_classes // 5))

    # Annotate the matrix cells with percentage values
    thresh = cm_normalized.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=max(8, 12 - num_classes // 5),
            )

    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)

    # Adjust layout with extra bottom margin for rotated labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2 + num_classes * 0.005)  # Dynamic bottom margin

    plt.savefig(output_path + f"conf_matrix.png")

    plt.clf()


def evaluate_model(model, test_loader, label_encoder, device, output_path, snr):
    """
    Evaluates the trained model and displays accuracy, confusion matrix, F1-score,
    and one output image per class.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test set.
        label_encoder: Label encoder to decode class names.
        device: 'cuda' or 'cpu' where evaluation happens.
    """
    model.to(device)  # Ensure model is on correct device
    model.eval()  # Set to evaluation mode

    y_true = []
    y_pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Model returns (output_image, output_class)
            output_class = model(inputs)

            # Get predicted class (argmax over logits)
            preds = torch.argmax(output_class, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().tolist())  # Move to CPU for metrics
            y_pred.extend(preds.cpu().tolist())

    # Compute Accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # Compute & Display Confusion Matrix
    class_names = label_encoder.classes_  # Decode label names
    cm = confusion_matrix(y_true, y_pred)
    # Normalize confusion matrix to percentages
    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100
    num_classes = len(class_names)

    # Plot confusion matrix
    fig, ax = plt.subplots(
        figsize=(max(10, num_classes * 0.8), max(8, num_classes * 0.6))
    )  # Dynamic size
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=class_names
    )
    disp.plot(
        cmap="Blues", values_format=".1f", ax=ax
    )  # Use 1 decimal place for percentages

    # Adjust x-axis label alignment and font sizes
    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
        va="top",
        fontsize=max(8, 12 - num_classes // 5),
    )
    ax.set_yticklabels(class_names, rotation=0, fontsize=max(8, 12 - num_classes // 5))
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix (Percentage)", fontsize=14)

    # Adjust layout with extra bottom margin for rotated labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2 + num_classes * 0.005)  # Dynamic bottom margin

    plt.savefig(output_path + f"{snr}_test_conf_matrix.png")

    plt.clf()

    return accuracy


def train_on_all_snrs(data_path, snr_range, mod_types, input_parameters, output_path):
    all_X_train = []
    all_y_train = []

    all_Xy_test = {}

    label_encoder = LabelEncoder()

    for snr in snr_range:
        input_data_folder = os.path.join(data_path, f"snr_{snr}")
        print(f"Loading {input_data_folder}")
        X, y = load_algorithm_snr_h5s(input_data_folder, mod_types)

        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42  # no test split
        )

        all_X_train.append(X_train)
        all_y_train.extend(y_train)

        all_Xy_test[snr] = {
            "X": X_test,
            "y": y_test,
        }

    # Stack all data
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.array(all_y_train)

    epoch_count = input_parameters["epoch_count"]
    learning_rate = input_parameters["learning_rate"]
    mds = "ALL" if len(mod_types) == 3 else mod_types[0]

    snrs = "ALL" if len(snr_range) == 13 else f"{snr_range[0]}_{snr_range[-1]}"

    output_data_folder = os.path.join(
        output_path, f"snr_{snrs}_mds_{mds}_e{epoch_count}_lr{learning_rate}\\"
    )
    os.makedirs(output_data_folder, exist_ok=True)

    joblib.dump(label_encoder, os.path.join(output_data_folder, "label_encoder.pkl"))

    # Prepare DataLoaders
    train_loader = prepare_dataloader(
        X_train,
        y_train,
        batch_size=32,
        shuffle=True,
    )

    model = MainModel(num_classes=len(np.unique(y_encoded))).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)

    start_time = time.time()

    # Train the model
    loss_history = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epoch_count,
        patience=50,
    )

    time_taken = time.time() - start_time

    # Save training metadata
    np.savetxt(
        os.path.join(output_data_folder, "loss_history.csv"),
        loss_history,
        delimiter=",",
    )
    plot_loss_curve(loss_history, output_data_folder)
    display_confusion_matrix(model, train_loader, device, output_data_folder)

    model_file_name = f"model_snr_{snrs}_mds_{mds}_e{epoch_count}_lr{learning_rate}.pth"
    torch.save(model.state_dict(), os.path.join(output_data_folder, model_file_name))

    # TEST

    for snr in snr_range:

        # Prepare DataLoaders
        test_loader = prepare_dataloader(
            all_Xy_test[snr]["X"],
            all_Xy_test[snr]["y"],
            batch_size=32,
        )

        # Evaluate the model
        acc = evaluate_model(
            model, test_loader, label_encoder, device, output_data_folder, snr
        )

        df = pd.read_csv(output_path + f"{input_parameters["csv"]}_results.csv")

        new_row = {
            "Algorithm": f"STFT-CNN_{snrs}",
            "SNR": snr,
            "Modulations": mds,
            "Accuracy (%)": acc,
            "Time Taken (Minutes)": time_taken,
            "Learning Rate": learning_rate,
            "Epoch Count": f"{len(loss_history)} / {epoch_count}",
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(output_path + f"{input_parameters["csv"]}_results.csv", index=False)


data_path = "C:\\Apps\\Code\\aimc-spec-7\\preprocessed_images\\stft\\"
output_path = "C:\\Apps\\Code\\STFT_CNN\\"

snr_range = [10, 5, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18, -20]

modulation_types = [
    "FM",
    # "PM",
]

input_parameters = {
    "epoch_count": 200,
    "learning_rate": 1e-3,
    "csv": "stft",
}

train_on_all_snrs(data_path, snr_range, modulation_types, input_parameters, output_path)
