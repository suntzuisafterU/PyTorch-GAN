import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train", include_B=True):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.include_B = include_B

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.jpg"))
        if include_B:
            self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.jpg"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.include_B:
            if self.unaligned:
                image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            else:
                image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if self.include_B:
            if image_B.mode != "RGB":
                image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        if self.include_B:
            item_B = self.transform(image_B)
            return {"A": item_A, "B": item_B}
        return {"A": item_A}

    def __len__(self):
        if self.include_B:
            return max(len(self.files_A), len(self.files_B))
        return len(self.files_A)

