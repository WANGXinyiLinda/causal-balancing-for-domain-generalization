# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import imp
from locale import normalize
import os
import torch
from collections import defaultdict
from torch.distributions.categorical import Categorical
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "ColoredMNIST_10class",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    # "WILDSCamelyon",
    # "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, dataset_transform, input_shape,
                 num_classes, hparams):
        super().__init__()
        self.hparams = hparams

        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        num_envs = len(self.hparams['environments'])
        for i in range(num_envs):
            images = original_images[i::num_envs]
            labels = original_labels[i::num_envs]
            self.datasets.append(dataset_transform(images, labels, self.hparams['environments'][i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root,
                                 self.color_dataset, (2, 28, 28,), 2, hparams)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(self.hparams['noise_rate'], len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class ColoredMNIST_10class(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST_10class, self).__init__(root,
                                 self.color_dataset, (10, 28, 28,), 10, hparams)

        self.input_shape = (10, 28, 28,)
        self.num_classes = 10

    def color_dataset(self, images, labels, environment):

        images = images.reshape((-1, 28, 28))

        # change label with prob 0.25
        prob_label = torch.ones((10, 10)).float() * (0.25 / 9)
        for i in range(10):
            prob_label[i, i] = 0.75

        labels_prob = torch.index_select(prob_label, dim=0, index=labels)
        labels = Categorical(probs=labels_prob).sample()

        # assign the color variable
        prob_color = torch.ones((10, 10)).float() * (environment / 9.0)
        for i in range(10):
            prob_color[i, i] = 1 - environment

        color_prob = torch.index_select(prob_color, dim=0, index=labels)
        color = Categorical(probs=color_prob).sample()

        # Apply the color to the image by zeroing out the other color channel
        output_images = torch.zeros((len(images), 10, 28, 28))

        idx_dict = defaultdict(list)
        for i in range(len(images)):
            idx_dict[int(labels[i])].append(i)
            output_images[i, color[i], :, :] = images[i]

        x = output_images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, 
                                self.rotate_dataset, (1, 28, 28,), 10, hparams)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        size = hparams["image_size"]
        augment = hparams['data_augmentation'] 
        normalize = hparams['normalize']

        if normalize:
            transform = transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor()
            ])
        
        if hparams['simple_augmentation']:
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, size, size,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams)


# class WILDSEnvironment:
#     def __init__(
#             self,
#             wilds_dataset,
#             metadata_name,
#             metadata_value,
#             transform=None):
#         self.name = metadata_name + "_" + str(metadata_value)

#         metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
#         metadata_array = wilds_dataset.metadata_array
#         subset_indices = torch.where(
#             metadata_array[:, metadata_index] == metadata_value)[0]

#         self.dataset = wilds_dataset
#         self.indices = subset_indices
#         self.transform = transform

#     def __getitem__(self, i):
#         x = self.dataset.get_input(self.indices[i])
#         if type(x).__name__ != "Image":
#             x = Image.fromarray(x)

#         y = self.dataset.y_array[self.indices[i]]
#         if self.transform is not None:
#             x = self.transform(x)
#         return x, y

#     def __len__(self):
#         return len(self.indices)


# class WILDSDataset(MultipleDomainDataset):
#     # INPUT_SHAPE = (3, 224, 224)
#     def __init__(self, dataset, metadata_name, test_envs, hparams):
#         super().__init__()
#         size = hparams["size"]
#         augment = hparams['data_augmentation']

#         transform = transforms.Compose([
#             transforms.Resize((size, size)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         augment_transform = transforms.Compose([
#             # transforms.Resize((size, size)),
#             transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         self.datasets = []

#         for i, metadata_value in enumerate(
#                 self.metadata_values(dataset, metadata_name)):
#             if augment and (i not in test_envs):
#                 env_transform = augment_transform
#             else:
#                 env_transform = transform

#             env_dataset = WILDSEnvironment(
#                 dataset, metadata_name, metadata_value, env_transform)

#             self.datasets.append(env_dataset)

#         self.input_shape = (3, size, size,)
#         self.num_classes = dataset.n_classes

#     def metadata_values(self, wilds_dataset, metadata_name):
#         metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
#         metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
#         return sorted(list(set(metadata_vals.view(-1).tolist())))


# class WILDSCamelyon(WILDSDataset):
#     ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
#             "hospital_4"]
#     def __init__(self, root, test_envs, hparams):
#         dataset = Camelyon17Dataset(root_dir=root)
#         super().__init__(
#             dataset, "hospital", test_envs, hparams)


# class WILDSFMoW(WILDSDataset):
#     ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
#             "region_4", "region_5"]
#     def __init__(self, root, test_envs, hparams):
#         dataset = FMoWDataset(root_dir=root)
#         super().__init__(
#             dataset, "region", test_envs, hparams)

