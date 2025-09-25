from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    """Place365 dataset."""

    def __init__(self, root_dir, data_dir, transform=None):
        """
        Args:
            root_dir (string): Path to recode-perceptions.
            data_dir (string): Directory with the images.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """

        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform

        self.samples = samples(root_dir, data_dir)

    def __len__(self):
        """returns total number of images in dataset"""

        return

    def __getitem__(self, idx):
        """returns next image in dataset"""

        path, label = self.samples[idx]
        # read image from path
        image = read_image(str(path))

        if self.transform:
            image = self.transform(image)

        return image, label


def samples(root_dir, data_dir):
    full_path = Path(root_dir) / Path(data_dir)
    classes = full_path.iterdir()
    class_to_idx = {cls.name: i for i, cls in enumerate(classes)}

    images = [
        (img, class_to_idx[img.parent.name])
        for img in full_path.rglob("*/*.jpg")
    ]
    return images