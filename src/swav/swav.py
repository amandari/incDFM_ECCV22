# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from PIL import ImageFilter


class SwAVWrapper:
    def __init__(self, backbone, num_backbone_features, device, nmb_crops, temperature=0.1, crops_for_assign=[0, 1], freeze_prototypes_niters=313, hidden_mlp=2048, output_dim=128, nmb_prototypes=3000):
        self.temperature = temperature
        self.nmb_crops = nmb_crops
        self.crops_for_assign = crops_for_assign
        self.freeze_prototypes_niters = freeze_prototypes_niters

        self.model = SwAVModule(
            backbone,
            num_backbone_features,  # resnet18: 512, resnet50: 2048
            hidden_mlp=hidden_mlp,
            output_dim=output_dim,
            nmb_prototypes=nmb_prototypes
        ).to(device)

        self.__init_optimizer()

    def __init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-6,
        )

    def train_step(self, batch, batch_idx):
        self.model.train()
        # normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = self.model(batch)
        embedding = embedding.detach()
        bs = batch[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # get assignments
                q = SwAVWrapper.sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q *
                                                F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        self.optimizer.zero_grad()
        loss.backward()

        # cancel gradients for the prototypes
        if batch_idx < self.freeze_prototypes_niters:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        self.optimizer.step()

        return loss.item()

    @staticmethod
    @torch.no_grad()
    def sinkhorn(out):
        SINKHORN_ITERATIONS = 3
        EPSILON = 0.05  # "regularization parameter for Sinkhorn-Knopp algorithm"
        WORLD_SIZE = 1
        # Q is K-by-B for consistency with notations from our paper
        Q = torch.exp(out / EPSILON).t()
        B = Q.shape[1] * WORLD_SIZE  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(SINKHORN_ITERATIONS):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class SwAVModule(nn.Module):
    def __init__(self, backbone: nn.Module, num_backbone_features, normalize=True, output_dim=128, hidden_mlp=2048, nmb_prototypes=3000):
        super().__init__()

        # backbone
        self.backbone = backbone

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_backbone_features, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_backbone_features, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(
                torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(
                self.radius_min, self.radius_max))
        )


class SwAVAugmenter:
    def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        color_transform = [
            self.get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = tr.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([tr.Compose([
                randomresizedcrop,
                tr.RandomHorizontalFlip(p=0.5),
                tr.Compose(color_transform),
                tr.ToTensor(),
                tr.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __call__(self, image):
        return list(map(lambda trans: trans(image), self.trans))

    def transform(self, image):
        return self.__call__(image)

    @staticmethod
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = tr.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = tr.RandomApply([color_jitter], p=0.8)
        rnd_gray = tr.RandomGrayscale(p=0.2)
        color_distort = tr.Compose([rnd_color_jitter, rnd_gray])
        return color_distort
