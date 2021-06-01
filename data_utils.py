import os
import PIL
import random
import tarfile
import smart_open
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset


class GenericDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.data_size = data.shape[0]

    def __getitem__(self, idx):
        img = transforms.ToPILImage()(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.data_size


class Imagenette(Dataset):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    data_folder = "imagenette2-160"
    internal_random_seed = 1234

    def __init__(
            self,
            root,
            download=False,
            train=True,
            transform=None
    ):
        self.root = root
        self._download(download)
        self.data_path = os.path.join(
            root, self.data_folder, "train" if train else "val")
        self.class_folders = sorted(
            f for f in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, f))
        )
        self.data = []
        self.targets = []
        self.data_size = 0
        self.train = train
        for i, fd in enumerate(self.class_folders):
            prefix = os.path.join(self.data_path, fd)
            self.data.extend(sorted(
                os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith("JPEG")))
            self.targets.extend(i for _ in range(len(self.data) - self.data_size))
            self.data_size = len(self.data)
        self._shuffle()  # shuffle the data with the preset internal random seed
        self.transform = transform

    def _shuffle(self):
        random.seed(self.internal_random_seed)
        random.shuffle(self.data)
        random.seed(self.internal_random_seed)
        random.shuffle(self.targets)

    def _download(self, download=False):
        if download:
            with smart_open.open(self.url, "rb") as file:
                with tarfile.open(fileobj=file, mode="r") as tgz:
                    tgz.extractall(self.root)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.data[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]

    def __len__(self):
        return self.data_size


def get_transforms(dataset="cifar10", augmentation=True):
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]) if augmentation else transforms.ToTensor()
        transform_test = transforms.ToTensor()
    elif dataset == "imagenette":
        transform_train = transforms.Compose([
            transforms.RandomCrop((128, 128), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]) if augmentation else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    else:
        raise NotImplementedError
    return transform_train, transform_test


def get_dataloaders(
        dataset,
        root,
        download,
        batch_size,
        augmentation=True,
        train_shuffle=True,
        num_workers=4
):
    if augmentation:
        transform_train, transform_test = get_transforms(dataset, True)
    else:
        transform_train, transform_test = get_transforms(dataset, False)
    if dataset == "cifar10":
        dataset_class = datasets.CIFAR10
    elif dataset == "imagenette":
        dataset_class = Imagenette
    else:
        raise NotImplementedError
    trainset = dataset_class(root=root, download=download, train=True, transform=transform_train)
    testset = dataset_class(root=root, download=download, train=False, transform=transform_test)
    trainloader = DataLoader(trainset, shuffle=train_shuffle, batch_size=batch_size, num_workers=num_workers)
    testloader = DataLoader(testset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return trainloader, testloader
