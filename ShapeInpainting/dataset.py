import numpy as np
import torch
from skimage import filters, measure
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def extract_landmarks_from_mnist(img, n_points=128, threshold=None):
    """
    Convert a MNIST grayscale image into a closed contour (landmarks).
    Returns Nx2 array of (x,y) points, uniformly sampled along the main contour.
    """

    if threshold is None:
        threshold = filters.threshold_otsu(img)
    mask = img > threshold

    contours = measure.find_contours(mask.astype(float), 0.5)

    if len(contours) == 0:
        return None

    contour = max(contours, key=len)

    d = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    t = np.linspace(0, total, n_points)
    x = np.interp(t, cum, contour[:, 1])
    y = np.interp(t, cum, contour[:, 0])
    pts = np.vstack([x, y]).T

    pts -= pts.mean(axis=0)
    pts /= np.linalg.norm(pts, axis=1).max()

    return pts


class MNISTShapesDataset(Dataset):
    def __init__(self, train=True, class_label=4, num_landmarks=128):

        self.class_label = class_label
        self.num_landmarks = num_landmarks

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((64, 64))]
        )
        dataset = datasets.MNIST(
            root=".", train=train, download=True, transform=transform
        )
        indices = [
            i for i, (_, label) in enumerate(dataset) if label == self.class_label
        ]
        self.dataset = torch.utils.data.Subset(dataset, indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx][0].squeeze().numpy()

        pts = extract_landmarks_from_mnist(np.flipud(img), n_points=self.num_landmarks)

        return torch.from_numpy(pts).float()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataset = MNISTShapesDataset(class_label=3, num_landmarks=64)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for idx, ax in enumerate(axes.flatten()):
        pts = dataset[idx]
        ax.plot(pts[:, 0], pts[:, 1], "-o")

    fig.suptitle("Landmarks (MNIST contour)")
    plt.show()
