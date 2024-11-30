import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np

from models.deeplabv3plus import DeepLabV3Plus
from preprocess import main_dataset  # Assuming the dataset class is from preprocess.py
import segmentation_models_pytorch as smp


# Argument parsing
parser = argparse.ArgumentParser(description="DeepLabV3Plus Network")
parser.add_argument("--data", type=str, default="/dataset", help="Path to dataset")
parser.add_argument("--weight", type=str, default="./saved_model/full_label_best.pth", help="Path to trained model weights")
parser.add_argument("--num-classes", type=int, default=20, help="Number of classes in the dataset")
parser.add_argument("--image-count", type=int, default=1000, help="Total number of images in the dataset")
parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
args = parser.parse_args()

print(args)

batch_size = 1  # Typically for evaluation, you might want a batch size of 1

# Data directories for the images and masks
images_dir = os.path.join(args.data, 'jpgs/rs19_val')
masks_dir = os.path.join(args.data, 'uint8_val')

# Use the main dataset class to load the data
org_dataset = main_dataset(
    images_dir,
    masks_dir,
    args.image_count,
    augmentation=None,  # No augmentation during evaluation
    preprocessing=None,  # Preprocessing already handled in model's get_preprocessing function
)

# Split dataset into training and validation
validation_split = args.val_split
val_dataset = torch.utils.data.Subset(org_dataset, range(int(args.image_count * (1 - validation_split)), args.image_count))

# DataLoader for the validation dataset
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False, pin_memory=True)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model and weights
net = DeepLabV3Plus(num_classes=args.num_classes)
net = net.to(device)

checkpoint = torch.load(args.weight)
net.load_state_dict(checkpoint['net'])

# Confusion matrix calculation
def get_confusion_matrix(gt_label, pred_label, class_num):
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

# Evaluation loop
def test():
    net.eval()
    
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    
    for idx, (images, labels) in enumerate(test_loader):
        _, h, w = labels.size()
        images = images.to(device)
        labels = labels.to(device)
        
        out = net(images)
        out = F.interpolate(out, size=(h, w), mode='bilinear')

        out = torch.argmax(out, dim=1)

        # Masking ignored class (19)
        ignore_index = labels != 19
        out = out[ignore_index]
        labels = labels[ignore_index]

        # Update confusion matrix
        confusion_matrix += get_confusion_matrix(labels.cpu().numpy(), out.cpu().numpy(), args.num_classes)

        print("\r[", idx ,"/", len(test_loader) ,"]", end='')
        sys.stdout.flush()

    # Compute per-class IoU
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    # Intersection over Union
    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    # Masked IoU (excluding ignore class)
    masked_arr = IU_array[IU_array != 0]
    masked_IU = masked_arr.mean()
    
    print("\nmIoU:", mean_IU)
    print('\nMasked IoU:', masked_IU)
    print(IU_array)

if __name__ == '__main__':
    test()
