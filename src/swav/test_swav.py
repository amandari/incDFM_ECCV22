from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from torchvision.datasets import ImageFolder, CIFAR10

from swav import SwAVWrapper, SwAVAugmenter

def main():
    # training 
    BATCH_SIZE = 32
    EPOCHS = 100

    # augmentation
    SIZE_CROPS = [224, 96]
    NMB_CROPS = [2, 4]
    MIN_SCALE_CROPS = [0.14, 0.9]
    MAX_SCALE_CROPS = [1, 1]
    
    augmenter = SwAVAugmenter(SIZE_CROPS, NMB_CROPS, MIN_SCALE_CROPS, MAX_SCALE_CROPS)
    
    train_dataset = ImageFolder("/media/zula/mvtec/capsule/train", transform=augmenter.transform)
    # train_dataset = CIFAR10(root='/data3/nahuja/cifar10', train=True, download=True, transform=augmenter.transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=20,
        # pin_memory=True,
        drop_last=True
    )

    # prep model
    backbone = torch.hub.load('pytorch/vision:v0.8.2', 'resnet50', pretrained=True)
    backbone.fc = nn.Identity()

    swav = SwAVWrapper(backbone, 2048, "cuda:0", NMB_CROPS)

    step = 0
    for epoch in tqdm(range(EPOCHS)):
        # TODO: Implement queue

        for it, (inputs, _) in enumerate(tqdm(train_loader)):
            swav_loss = swav.train_step(inputs, step)
            tqdm.write(str(swav_loss))
            step += 1


if __name__ == "__main__":
    main()
