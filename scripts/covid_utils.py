# Standard library imports
import os
import random
import zipfile
from collections import Counter

# Data science and image processing
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.image import imread

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.metrics import MeanIoU

# Third-party libraries
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score

# Google Colab specific imports
from google.colab import drive, runtime

# Other utilities
import kagglehub

project_dir='/content/drive/MyDrive/Covid-19_Presence'
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels, sources) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate running accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (preds == labels).sum().item()
        total_samples += images.size(0)

        # Print progress every 20 batches with current running accuracy
        if batch_idx % 20 == 0:
            current_acc = correct_predictions / total_samples
            avg_loss = running_loss / total_samples
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {avg_loss:.4f}, Running Acc: {current_acc:.4f}')

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def val_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels, sources in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Fixed the comparison here too
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def create_metadata_df(dir,):
  labels=[]
  image_dirs=[]
  inf_mask_dirs=[]
  fnames=[]
  lung_mask_dirs=[]
  for subdir in ['Non-COVID', 'Normal', 'COVID-19']:

    for file in os.listdir(os.path.join(dir, subdir, 'images')):
      labels.append(subdir)
      fnames.append(file)
      image_dirs.append(os.path.join(dir, subdir, 'images'))
      inf_mask_dirs.append(os.path.join(dir, subdir, 'infection masks'))
      lung_mask_dirs.append(os.path.join(dir, subdir, 'lung masks'))
  return pd.DataFrame({'filename':fnames,'image_directory':image_dirs,'infection_mask_dir':inf_mask_dirs,'label':labels,'lung_mask_dir':lung_mask_dirs})



def plot_image_and_mask(image_path, mask_path, title='', save_plot=False, save_dir='.'):
    """
    Loads and displays an image, its infection mask, and an overlay side-by-side,
    with an option to save the plot.
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert BGR image to RGB for correct color display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the mask as a grayscale image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to load mask at {mask_path}")
        return

    # Normalize the mask for the overlay (values will be 0 or 1)
    mask_norm = mask / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image {title}')
    axes[0].axis('off')

    # Plot the infection mask
    axes[1].imshow(mask, cmap='viridis')
    axes[1].set_title(f'Infection Mask: {title}')
    axes[1].axis('off')

    # Plot the overlay
    axes[2].imshow(image)
    axes[2].imshow(mask_norm, cmap='Reds', alpha=0.5)
    axes[2].set_title(f'Overlay {title}')
    axes[2].axis('off')

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'image_and_mask_overlay_{os.path.basename(image_path)}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

def calculate_stats_from_df(df):
  """
  Calculates mean and standard deviation directly from a DataFrame
  of image file paths after resizing to 512x512.
  """
  all_pixels = []

  for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Stats"):
      img_path = os.path.join(row['image_directory'], row['filename'])

      # Read the image in grayscale (0-255)
      image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

      if image is not None:
          # --- CRITICAL FIX: Resize the image before calculating stats ---
          image = cv2.resize(image, (512, 512))

          # Flatten the pixel values and add to our list
          all_pixels.extend(image.flatten())

  pixels = np.array(all_pixels, dtype=np.float32)

  # Calculate stats on the raw pixel values (0-255)
  mean = np.mean(pixels)
  std = np.std(pixels)

  print("\n✅ Calculation complete.")
  print(f"Raw Mean (0-255 range): {mean:.4f}")
  print(f"Raw Std (0-255 range): {std:.4f}")

  # These are the values to use for your A.Normalize transform
  normalized_mean = mean / 255.0
  normalized_std = std / 255.0

  print(f"Normalized Mean (0-1 range): {normalized_mean:.4f}")
  print(f"Normalized Std (0-1 range): {normalized_std:.4f}")

  return normalized_mean, normalized_std


def plot_masked_images_and_masks(dataset, num_to_plot=4, save_plot=False, save_dir='.'):
  """
  Plots a few images from the dataset along with their infection masks and optionally saves the plot.
  """
  if num_to_plot == 1:
      fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  else:
      fig, axes = plt.subplots(num_to_plot, 2, figsize=(10, num_to_plot * 5))

  # Denormalization values to get the image back to a viewable range
  mean = 0.5300
  std = 0.2447

  for i in range(num_to_plot):
      # Get a sample from the dataset
      image, infection_mask = dataset[i]

      # Denormalize the image tensor for plotting
      image_denorm = (image.squeeze().numpy() * std) + mean

      # Convert mask to numpy
      infection_mask_np = infection_mask.squeeze().numpy()

      # Handle different axes indexing for num_to_plot = 1 vs > 1
      if num_to_plot == 1:
          ax1 = axes[0]
          ax2 = axes[1]
      else:
          ax1 = axes[i, 0]
          ax2 = axes[i, 1]

      # Plot the masked image
      ax1.imshow(image_denorm, cmap='gray')
      ax1.set_title(f"Masked Image Sample {i+1}")
      ax1.axis('off')

      # Plot the infection mask
      ax2.imshow(infection_mask_np, cmap='viridis')
      ax2.set_title(f"Infection Mask Sample {i+1}")
      ax2.axis('off')

  plt.tight_layout()

  if save_plot:
      os.makedirs(save_dir, exist_ok=True)
      filename = f'masked_images_and_masks_{num_to_plot}_samples.png'
      save_path = os.path.join(save_dir, filename)
      plt.savefig(save_path)
      print(f"Plot saved to {save_path}")

  plt.show()


def combined_loss(y_pred, y_true):
    return dice_loss(y_pred, y_true) + bce_loss(y_pred, y_true)

# --- Training and Validation Functions (same as your code) ---
def calculate_iou(y_pred, y_true, threshold=0.5):
    # Normalize ground truth mask to 0 or 1
    y_true_binary = (y_true > 0).float()

    # Binarize the predicted mask
    y_pred_binary = (y_pred > threshold).float()

    # Calculate intersection and union
    intersection = (y_pred_binary * y_true_binary).sum()
    union = y_pred_binary.sum() + y_true_binary.sum() - intersection

    # Return IoU
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()

# Training function
def train_epoch_segmentation(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += calculate_iou(outputs, masks)

        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    return total_loss / len(train_loader), total_iou / len(train_loader)

def validate_epoch_segmentation(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            if images is None: continue # Skip None batches
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += calculate_iou(outputs, masks)
    return total_loss / len(val_loader), total_iou / len(val_loader)

def visualize_val_predictions(model, val_loader, device, num_samples=6):
    """
    Visualizes original images, ground truth masks, and predicted masks
    from the validation DataLoader.
    """
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Denormalization constants from your dataset class
    mean = torch.tensor([0.5300], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor([0.2447], dtype=torch.float32, device=device).view(1, -1, 1, 1)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            if idx * val_loader.batch_size >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)

            # Get prediction
            preds = model(images)
            preds = torch.sigmoid(preds)

            for i in range(images.size(0)):
                if (idx * val_loader.batch_size) + i >= num_samples:
                    break

                # Convert to numpy for plotting, and denormalize
                img_np = (images[i] * std + mean).squeeze().cpu().numpy()
                mask_np = masks[i].squeeze().cpu().numpy()
                pred_np = preds[i].squeeze().cpu().numpy()

                # Plot image
                axes[(idx * val_loader.batch_size) + i, 0].imshow(img_np, cmap='gray')
                axes[(idx * val_loader.batch_size) + i, 0].set_title(f'Sample {i+1}: Original (Noisy) Image')
                axes[(idx * val_loader.batch_size) + i, 0].axis('off')

                # Plot ground truth mask
                axes[(idx * val_loader.batch_size) + i, 1].imshow(mask_np, cmap='gray')
                axes[(idx * val_loader.batch_size) + i, 1].set_title('Ground Truth Mask')
                axes[(idx * val_loader.batch_size) + i, 1].axis('off')

                # Plot prediction
                iou_score = calculate_iou_np(pred_np, mask_np)
                axes[(idx * val_loader.batch_size) + i, 2].imshow(pred_np, cmap='gray')
                axes[(idx * val_loader.batch_size) + i, 2].set_title(f'Predicted Mask (IoU: {iou_score:.3f})')
                axes[(idx * val_loader.batch_size) + i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_iou_np(y_pred, y_true, threshold=0.5):
    """
    Calculates the IoU (Jaccard score) for numpy arrays.
    """
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true_binary = (y_true > 0).astype(np.float32)
    intersection = (y_pred_binary * y_true_binary).sum()
    union = y_pred_binary.sum() + y_true_binary.sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou


def validate_lung_cropping(dataset, num_samples=6, save_plot=False, save_dir='.'):
    """
    Compare original images with lung-cropped versions and optionally save the plot.
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))

    for i in range(num_samples):
        # Get a sample
        row = dataset.df.iloc[i]
        img_path = os.path.join(row['Image_Directory'], row['filename'])

        # Load original image
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Get lung crop bbox
        bbox = dataset.get_lung_crop_bbox(img_path, original.shape)

        # Get cropped version
        if bbox is not None:
            y_min, y_max, x_min, x_max = bbox
            cropped = original[y_min:y_max, x_min:x_max]
            crop_title = f"Cropped {cropped.shape}"
        else:
            cropped = original
            crop_title = "No lung detected"

        # Plot original
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f"Original {original.shape}")
        axes[0, i].axis('off')

        # Plot cropped
        axes[1, i].imshow(cropped, cmap='gray')
        axes[1, i].set_title(crop_title)
        axes[1, i].axis('off')

        # Draw bounding box on original
        if bbox is not None:
            y_min, y_max, x_min, x_max = bbox
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[0, i].add_patch(rect)

    axes[0, 0].set_ylabel('Original', fontsize=14)
    axes[1, 0].set_ylabel('Lung Cropped', fontsize=14)
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'lung_cropping_check_{num_samples}_samples.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

def validate_with_transforms(dataset, num_samples=4, save_plot=False, save_dir='.'):
    """
    Show how images look after transforms (final dataset output) and optionally save the plot.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))

    # Denormalization values
    mean = torch.tensor([0.5300])
    std = torch.tensor([0.2447])

    for i in range(num_samples):
        # Get processed sample from dataset - NOW RETURNS 3 ITEMS
        image, label, source = dataset[i]  # Updated to unpack 3 items

        # Denormalize for plotting
        image_denorm = image * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        img_np = image_denorm.squeeze().numpy()

        # Plot with source information
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(f'Label: {label.item():.0f}\nSource: {source}')  # Added source to title
        axes[i].axis('off')

    plt.suptitle('Final Processed Images (After Transforms)')
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'processed_images_check_{num_samples}_samples.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

def validate_with_transforms_detailed(dataset, num_samples=4, save_plot=False, save_dir='.'):
    """
    Show more detailed information including source distribution and save the plot.
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))

    # Denormalization values
    mean = torch.tensor([0.5300])
    std = torch.tensor([0.2447])

    sources_found = []

    for i in range(num_samples):
        # Get processed sample from dataset
        image, label, source = dataset[i]
        sources_found.append(source)

        # Denormalize for plotting
        image_denorm = image * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        img_np = image_denorm.squeeze().numpy()

        # Plot processed image
        axes[0, i].imshow(img_np, cmap='gray')
        axes[0, i].set_title(f'Processed\nLabel: {label.item():.0f}')
        axes[0, i].axis('off')

        # Get and plot original for comparison
        row = dataset.df.iloc[i]
        img_path = os.path.join(row['Image_Directory'], row['filename'])
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        axes[1, i].imshow(original, cmap='gray')
        axes[1, i].set_title(f'Original\nSource: {source}')
        axes[1, i].axis('off')

    source_counts = Counter(sources_found)
    print(f"Source distribution in sample: {dict(source_counts)}")

    axes[0, 0].set_ylabel('After Processing', fontsize=12)
    axes[1, 0].set_ylabel('Original', fontsize=12)
    plt.suptitle('Comparison: Original vs Processed Images')
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'data_transforms_check_{num_samples}_samples.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

def quick_lung_mask_check(dataset, idx=0, save_plot=False, save_dir='.'):
    """
    Check if lung segmentation is working on a single image and optionally save the plot.
    """
    row = dataset.df.iloc[idx]
    img_path = os.path.join(row['Image_Directory'], row['filename'])

    # Load original
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Get lung mask (for visualization)
    img_256 = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
    img_arr = tf.keras.preprocessing.image.img_to_array(img_256).squeeze() / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    mask_pred = dataset.lung_model.predict(img_arr, verbose=0)
    mask = (mask_pred.squeeze() > 0.5).astype(np.uint8)

    # Resize mask back to original size
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f'Original Image\nSource: {row["source"]}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_resized, cmap='gray')
    plt.title('Lung Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original, cmap='gray', alpha=0.7)
    plt.imshow(mask_resized, cmap='Reds', alpha=0.3)
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()

    if save_plot:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        filename = f'lung_mask_check_idx_{idx}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()

def check_dataloader_with_sources(dataloader, num_batches=2):
    """
    New function to validate the dataloader with source information
    """
    print("Checking DataLoader with source information:")
    print("-" * 50)

    for batch_idx, (images, labels, sources) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sources: {sources}")
        print(f"  Unique sources in batch: {set(sources)}")

        # Count labels
        pos_count = (labels == 1).sum().item()
        neg_count = (labels == 0).sum().item()
        print(f"  Label distribution - Positive: {pos_count}, Negative: {neg_count}")

        # Source distribution
        from collections import Counter
        source_dist = Counter(sources)
        print(f"  Source distribution: {dict(source_dist)}")
        print()

def patient_split(all_data, val_size=0.2, test_size=0.2, random_state=42):
  all_patients = all_data['patient_id'].unique()

  # First split into train+val and test
  train_val_patients, test_patients = train_test_split(
      all_patients, test_size=test_size, random_state=random_state)

  # Then split train+val into train and val
  train_patients, val_patients = train_test_split(
      train_val_patients, test_size=val_size, random_state=random_state)

  train_data = all_data[all_data['patient_id'].isin(train_patients)]
  val_data = all_data[all_data['patient_id'].isin(val_patients)]
  test_data = all_data[all_data['patient_id'].isin(test_patients)]

  return train_data, val_data, test_data

def print_source_balance(train_data,val_data,test_data):
  print('*******Training**********')
  positives=train_data[train_data['label']=='positive']
  negatives=train_data[train_data['label']=='negative']
  print(f"Positive source value counts {positives['source'].value_counts()}")
  print(f"Negative source value counts {negatives['source'].value_counts()}")
  print()
  print('*******Validation**********')

  positives=val_data[val_data['label']=='positive']
  negatives=val_data[val_data['label']=='negative']
  print(f"Positive source value counts {positives['source'].value_counts()}")
  print(f"Negative source value counts {negatives['source'].value_counts()}")
  print()
  print('*******Testing**********')

  positives=test_data[test_data['label']=='positive']
  negatives=test_data[test_data['label']=='negative']
  print(f"Positive source value counts {positives['source'].value_counts()}")
  print(f"Negative source value counts {negatives['source'].value_counts()}")
  print()

def visualize_inference_on_test_set(model, df, transform, device, num_samples=6):

    model.eval() # Set the model to evaluation mode

    # Create a simple dataset without masks
    class InferenceDataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(row['Image_Directory'], row['filename'])
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Albumentations requires both image and mask, so we use a dummy mask
            dummy_mask = np.zeros_like(image)

            if self.transform:
                augmented = self.transform(image=image, mask=dummy_mask)
                image = augmented['image']

            return image

    test_dataset = InferenceDataset(df, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx, image in enumerate(test_loader):
            if idx >= num_samples:
                break

            image = image.to(device)
            pred = model(image)

            # Apply a threshold to the predicted mask for visualization
            pred_binary = (pred > 0.5).float()

            # Convert to numpy for plotting
            img_np = image[0, 0].cpu().numpy()
            pred_np = pred_binary[0, 0].cpu().numpy()

            # Plot the results
            axes[idx, 0].imshow(img_np, cmap='gray')
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(pred_np, cmap='gray')
            axes[idx, 1].set_title('Predicted Mask')
            axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

def show_predictions_and_masks(classifier_model, segmentation_model, val_loader, device, num_images=4, save_plot=False, project_dir='.'):
    classifier_model.eval()  # Set classifier to evaluation mode
    segmentation_model.eval() # Set segmentation model to evaluation mode

    images_shown = 0

    # Denormalization values (assuming you used these for training)
    mean = torch.tensor([0.5300], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor([0.2447], dtype=torch.float32, device=device).view(1, -1, 1, 1)

    with torch.no_grad():
        for i, (images, labels, sources) in enumerate(val_loader):
            if images_shown >= num_images:
                break

            images, labels = images.to(device), labels.to(device)

            # --- Classification Prediction ---
            classifier_outputs = classifier_model(images)
            classifier_probs = torch.sigmoid(classifier_outputs)
            classification_predictions = (classifier_probs > 0.5).long().squeeze(1)

            # --- Segmentation Prediction ---
            segmentation_outputs = segmentation_model(images)
            predicted_masks = (segmentation_outputs > 0.5).float() # Binarize the mask

            for j in range(len(images)):
                if images_shown >= num_images:
                    break

                # Original image (denormalized)
                original_image = (images[j] * std + mean).cpu().squeeze().numpy()

                # True label and predicted classification
                true_label_str = 'COVID-19' if labels[j].item() == 1 else 'Not COVID-19'
                predicted_label_str = 'COVID-19' if classification_predictions[j].item() == 1 else 'Not COVID-19'

                # Predicted mask
                mask_to_plot = predicted_masks[j].cpu().squeeze().numpy()

                # Create the plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # Two subplots: image and mask
                source = sources[j]

                # Plot Original Image
                axes[0].imshow(original_image, cmap='gray')
                axes[0].set_title(f'True: {true_label_str}\nPred: {predicted_label_str}\nSource: {source}', fontsize=14)
                axes[0].axis('off')

                # Plot Predicted Mask
                axes[1].imshow(mask_to_plot, cmap='viridis', alpha=0.7) # Using viridis for mask for visibility
                axes[1].set_title('Predicted Infection Mask', fontsize=14)
                axes[1].axis('off')

                plt.tight_layout()

                if save_plot:
                    # Create the filename
                    filename = f'mask_and_pred_{j}.png'
                    save_path = os.path.join(project_dir, filename)

                    # Ensure the directory exists
                    os.makedirs(project_dir, exist_ok=True)

                    plt.savefig(save_path)
                    print(f"Plot saved to {save_path}")

                plt.show()

                images_shown += 1
def load_best_classifier(model, models_dir, device='cuda',relative_path_classifier='best_classifier_model3.pth'):
  """
  Load the best classifier model from checkpoint

  Args:
      model: The model architecture (should be initialized)
      models_dir: Directory where the model is saved
      device: Device to load the model on

  Returns:
      model: Loaded model
      checkpoint_info: Dictionary with training info
  """

  checkpoint_path = os.path.join(models_dir, relative_path_classifier)

  # Check if file exists
  if not os.path.exists(checkpoint_path):
      raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

  # Load checkpoint
  checkpoint = torch.load(checkpoint_path, map_location=device)

  # Load model state
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)

  # Print checkpoint info
  print(f"Model loaded successfully!")
  print(f"Best validation accuracy: {checkpoint['best_accuracy']:.4f}")
  print(f"Saved at epoch: {checkpoint['epoch'] + 1}")

  # Return model and checkpoint info
  checkpoint_info = {
      'best_accuracy': checkpoint['best_accuracy'],
      'epoch': checkpoint['epoch'],
      'optimizer_state_dict': checkpoint.get('optimizer_state_dict', None)
  }

  return model, checkpoint_info

def evaluation_with_source(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_sources = []

    with torch.no_grad():
        for images, labels, sources in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_sources.extend(sources)

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss, all_preds, all_labels, all_sources

def simple_gradcam_analysis(model, image_tensor, original_image, lung_mask=None):
    """
    Simple Grad-CAM implementation that works with any model architecture
    """
    model.eval()
    
    # Enable gradients for input
    image_tensor.requires_grad_(True)
    
    # Forward pass
    output = model(image_tensor)
    
    # Get prediction
    prob = torch.sigmoid(output).item()
    pred = "COVID-19" if prob > 0.5 else "Non-COVID"
    
    # Backward pass
    model.zero_grad()
    output[0, 0].backward()
    
    # Get gradients of input image
    gradients = image_tensor.grad.data.abs()
    
    # Create simple gradient-based attention map
    attention = gradients.squeeze().cpu().numpy()
    
    # Resize attention to match original image
    h, w = original_image.shape
    attention_resized = cv2.resize(attention, (w, h))
    
    # Normalize
    attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Preprocessed image (what model sees)
    processed_img = image_tensor.detach().squeeze().cpu().numpy()
    # Denormalize for visualization
    mean, std = 0.5300, 0.2447
    processed_img = processed_img * std + mean
    axes[1].imshow(processed_img, cmap='gray')
    axes[1].set_title('Model Input (Masked)')
    axes[1].axis('off')
    
    # Attention heatmap
    axes[2].imshow(attention_resized, cmap='jet')
    axes[2].set_title('Attention Map')
    axes[2].axis('off')
    
    # Overlay on original
    axes[3].imshow(original_image, cmap='gray', alpha=0.7)
    axes[3].imshow(attention_resized, cmap='jet', alpha=0.5)
    if lung_mask is not None:
        # Show lung boundaries
        contours, _ = cv2.findContours(lung_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            axes[3].plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=1)
    axes[3].set_title('Attention + Lung Boundaries')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Prediction: {pred} (probability: {prob:.3f})")
    return attention_resized

def analyze_sample_simple(model, dataset, idx=0):
    """Analyze a specific sample from your dataset using simple gradient-based attention"""
    # Get sample
    row = dataset.df.iloc[idx]
    img_path = os.path.join(row['Image_Directory'], row['filename'])
    
    # Load original
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Get processed version from dataset
    processed_img, label, source = dataset[idx]
    processed_tensor = processed_img.unsqueeze(0).to(device)
    
    # Get lung mask for visualization
    img_256 = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
    img_arr = tf.keras.preprocessing.image.img_to_array(img_256).squeeze() / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    
    mask_pred = dataset.lung_model.predict(img_arr, verbose=0)
    mask = (mask_pred.squeeze() > 0.5).astype(np.uint8)
    lung_mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    print(f"Analyzing sample {idx}: {row['filename']}")
    print(f"True label: {label.item():.0f}, Source: {source}")
    
    # Visualize
    attention = simple_gradcam_analysis(model, processed_tensor, original, lung_mask)
    
    return attention

# Alternative: Hook-free approach that works with any architecture
def integrated_gradients_analysis(model, image_tensor, original_image, steps=50):
    """
    Integrated gradients approach - more robust than Grad-CAM
    """
    model.eval()
    
    # Create baseline (noise background similar to your training)
    baseline = torch.randn_like(image_tensor) * 0.2447 + 0.5300  # Same as your noise
    
    # Generate path from baseline to image
    alphas = torch.linspace(0, 1, steps)
    gradients = []
    
    for alpha in alphas:
        # Interpolated image
        interpolated = baseline + alpha * (image_tensor - baseline)
        interpolated.requires_grad_(True)
        
        # Forward and backward
        output = model(interpolated)
        model.zero_grad()
        output[0, 0].backward()
        
        gradients.append(interpolated.grad.data.clone())
        
    # Average gradients and multiply by image difference
    avg_gradients = torch.mean(torch.stack(gradients), dim=0)
    integrated_grad = (image_tensor - baseline) * avg_gradients
    
    # Convert to attention map
    attention = integrated_grad.squeeze().abs().cpu().numpy()
    
    # Resize and normalize
    h, w = original_image.shape
    attention_resized = cv2.resize(attention, (w, h))
    attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    
    return attention_resized

def visualize_random_val_prediction(model, val_loader, device, save_plot=False, save_dir='.'):
    """
    Visualizes a random original image, ground truth mask, and predicted mask
    from the validation DataLoader and optionally saves the plot.
    """
    model.eval()
    # Get total number of samples in validation dataset
    total_samples = len(val_loader.dataset)
    # Choose a random sample index
    rand_idx = random.randint(0, total_samples - 1)
    # Determine batch index and in-batch index
    batch_size = val_loader.batch_size
    batch_idx = rand_idx // batch_size
    in_batch_idx = rand_idx % batch_size

    # Denormalization constants for plotting
    mean = torch.tensor([0.5300], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor([0.2447], dtype=torch.float32, device=device).view(1, -1, 1, 1)

    # Fetch the batch containing the random sample
    for idx, (images, masks) in enumerate(val_loader):
        if idx == batch_idx:
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(images))
            break

    # Denormalize and convert tensors to numpy arrays for plotting
    img_np = (images[in_batch_idx] * std + mean).squeeze().cpu().numpy()
    mask_np = masks[in_batch_idx].squeeze().cpu().numpy()
    pred_np = preds[in_batch_idx].squeeze().detach().cpu().numpy()

    iou_score = calculate_iou_np(pred_np, mask_np)  # Ensure this function is defined

    # Plot images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original (Noisy) Image')
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title(f'Predicted Mask (IoU: {iou_score:.3f})')
    axes[2].axis('off')

    plt.tight_layout()

    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'random_val_prediction_{rand_idx}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()


def retrieve_and_process_data(zip_filename='COVID-19_Radiography_Dataset.zip'):
    """
    Retrieves files from a zip file in Google Drive, and extracts all its content.
    It handles potential errors during the process and prints informative messages.
    Args:
        zip_filename (str, optional): The name of the zip file in Google Drive.
            Defaults to 'COVID-19_Radiography_Dataset.zip'.
    Returns:
        None:  The function extracts files to a directory.
    """
    try:
        # Mount Google Drive
        drive.mount('/content/drive')
        
        # Updated path to zip file
        zip_filepath = f'/content/drive/My Drive/Covid-19_Presence/{zip_filename}'
        
        # Check if the zip file exists
        if not os.path.exists(zip_filepath):
            print(f"Error: Zip file not found at {zip_filepath}. Please ensure the file is in your Google Drive.")
            return None
        
        # Extract the file
        extraction_path = '/content/data'  # Directory to extract to
        os.makedirs(extraction_path, exist_ok=True)  # Make the directory
        
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            try:
                zf.extractall(extraction_path)  # Extract *all* files
                print(f"Successfully extracted all files from {zip_filename} to {extraction_path}")
            except Exception as e:
                print(f"Error extracting files: {e}")
                return None
                
    except Exception as e:
        print(f"Error mounting drive or processing zip: {e}")
        return None

