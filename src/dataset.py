import torchvision
import torchvision.transforms as transforms

def get_transform(dataset_name, image_range="-1,1"):
    if dataset_name in ["MNIST", "FashionMNIST"]:
        mean = (0.5,)
        std = (0.5,)
    elif dataset_name == "CIFAR10":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError(dataset_name)

    if image_range == "0,1":
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    elif image_range == "-1,1":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    else:
        raise ValueError(image_range)


def get_dataset(dataset_name, root="./data", train=True, image_range="-1,1"):
    # データの前処理のためのtransformを取得します。
    transform = get_transform(dataset_name, image_range)

    if dataset_name == "MNIST":
        return torchvision.datasets.MNIST(
            root=root / "data_mnist", train=train, download=True, transform=transform
        )

    elif dataset_name == "FashionMNIST":
        return torchvision.datasets.FashionMNIST(
            root=root / "data_fashion_mnist", train=train, download=True, transform=transform
        )

    elif dataset_name == "CIFAR10":
        return torchvision.datasets.CIFAR10(
            root=root / "data_cifar10", train=train, download=True, transform=transform
        )

    else:
        raise ValueError(dataset_name)