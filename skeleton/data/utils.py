import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid


def print_image_from_dataloader(dataloader, index):
    batch = next(iter(dataloader))  # Get a batch of images from the data loader
    images = batch[index]  # Select the images at the specified index from the batch

    # Convert the tensor images to a grid and transpose the dimensions
    grid = make_grid(images, nrow=8, normalize=True)
    image_np = TF.to_pil_image(grid)

    # Display the image using matplotlib
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()
