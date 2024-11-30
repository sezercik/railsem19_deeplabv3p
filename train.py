import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
import albumentations as album
from tensorboardX import SummaryWriter
import os
import sys
import argparse

from models.deeplabv3plus import DeepLabV3Plus
import segmentation_models_pytorch as smp
from preprocess import main_dataset


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=720, width=720, always_apply=True),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x):
    if x.ndim == 3:  # If the array is an image with 3 dimensions
        return x.transpose(2, 0, 1).astype('float32')  # Convert image (H, W, C) to (C, H, W)
    elif x.ndim == 2:  # If the array is a mask with 2 dimensions (H, W)
        return x.astype('long')  # Mask should be of type long (int64) for PyTorch loss functions
    else:
        raise ValueError(f"Unsupported array dimensions: {x.ndim}")



def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)


# Argument parser
parser = argparse.ArgumentParser(description="DeepLabV3Plus Network")
parser.add_argument("--data", type=str, default="/dataset", help="")
parser.add_argument("--batch-size", type=int, default=4, help="")
parser.add_argument("--worker", type=int, default=4, help="")
parser.add_argument("--epoch", type=int, default=200, help="")
parser.add_argument("--num-classes", type=int, default=19, help="")
parser.add_argument("--momentum", type=float, default=0.9, help="")
parser.add_argument("--lr", type=float, default=1e-2, help="")
parser.add_argument("--os", type=int, default=16, help="")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="")
parser.add_argument("--logdir", type=str, default="./logs/", help="")
parser.add_argument("--save", type=str, default="./saved_model/", help="")
parser.add_argument("--image_count", type=int, default=8500, help="How much image you want to train")
parser.add_argument('-v', '--val_split', type=float, default=0.3, help='Validation Split for training and validation set.')

args = parser.parse_args()

# Set the device to use the available GPUs (Kaggle has at least two GPUs available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)  # Set the first GPU as the default (you can specify a different GPU here)

# Data loading
images_dir = os.path.join(args.data, 'jpgs/rs19_val')
masks_dir = os.path.join(args.data, 'uint8/rs19_val')
jsons_dir = os.path.join(args.data, 'jsons/rs19_val')

org_dataset = main_dataset(
    images_dir,
    masks_dir,
    args.image_count,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(smp.encoders.get_preprocessing_fn('resnet101', 'imagenet')),
)

# Split dataset into training and validation sets
validation_split = args.val_split
train_dataset, val_dataset = random_split(org_dataset, [int(args.image_count * (1 - args.val_split)),
                                                        int(args.image_count * args.val_split)])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.worker, drop_last=False, shuffle=True, pin_memory=True)

# Initialize the model
net = DeepLabV3Plus(num_classes=args.num_classes, os=args.os)
net = net.to(device)

# Use DataParallel for multi-GPU setup
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)  # This will wrap the model for multi-GPU usage

# Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-4)

# Tensorboard writer
writer = SummaryWriter(args.logdir)


def train(epoch, iteration, scheduler, total_loss):
    epoch += 1
    net.train()

    train_loss = 0
    for idx, (images, labels) in enumerate(train_loader):
        iteration += 1
        _, _, h, w = labels.size()  # Unpack batch_size, num_classes, height, width

        images, labels = images.to(device), labels.to(device).long()
        out = net(images)
        out = F.interpolate(out, size=(h, w), mode='bilinear')

        loss = criterion(out, labels)

        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\repoch: {epoch}, iter: {idx + 1}/{len(train_loader)}, loss: {loss.item():.4f}", end='')
        sys.stdout.flush()

    scheduler.step()

    # Log to TensorBoard
    writer.add_scalar('log/loss', train_loss / (idx + 1), epoch)
    writer.add_scalar('log/lr', scheduler.get_lr()[0], epoch)

    print(f"\nepoch: {epoch}, loss: {train_loss / (idx + 1):.4f}, lr: {scheduler.get_lr()[0]:.6f}")

    # Save the best model
    if epoch == args.epoch:
        saving_path = os.path.join(args.save, 'last_model.pth')
        torch.save(net.state_dict(), saving_path)
        print(f"Model saved in {saving_path}")

    return epoch, iteration, total_loss


# Training loop
if __name__ == '__main__':
    epoch = 0
    iteration = 0
    total_loss = 1e9

    while epoch < args.epoch:
        epoch, iteration, total_loss = train(epoch, iteration, scheduler, total_loss)

    print("Training finished!")