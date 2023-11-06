import torch
import random
import os
import shutil
from PIL import Image
import numpy as np
from torchvision import transforms


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if (
                self.num_imgs < self.pool_size
            ):  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if (
                    p > 0.5
                ):  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(
                        0, self.pool_size - 1
                    )  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


def grayscale_to_rgb(grayscale_image):
    """Converts a tensor grayscale image to an RGB image."""
    rgb_image = torch.zeros((3, grayscale_image.shape[1], grayscale_image.shape[2]))
    for i in range(3):
        rgb_image[i] = grayscale_image

    if grayscale_image.is_cuda:
        rgb_image = rgb_image.cuda()

    return rgb_image


def create_folder(folder_path, overwrite=False):
    try:
        if os.path.exists(folder_path):
            if overwrite:
                shutil.rmtree(folder_path)  # Remove the folder and its content
                os.makedirs(folder_path)  # Recreate the folder
                print(f"Folder '{folder_path}' created and content overwritten.")
            else:
                print(f"Folder '{folder_path}' already exists, content not modified.")
        else:
            os.makedirs(folder_path)  # Create the folder
            print(f"Folder '{folder_path}' created.")
    except Exception as e:
        print(f"Error creating or modifying folder: {e}")


def load_images_from_folder(folder_path, image_mode="RGB", as_tensor=True):
    image_list = []

    # Determine the image mode (RGB or grayscale)
    if image_mode == "RGB":
        mode = "RGB"
    elif image_mode == "grayscale":
        mode = "L"
    else:
        raise ValueError("Invalid image_mode. Use 'RGB' or 'grayscale'.")

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path).convert(mode)

            if as_tensor:
                # Convert to PyTorch tensor
                image = transforms.ToTensor()(image)
                image_list.append(image)
            else:
                # Convert to NumPy array
                image = np.array(image)
                image_list.append(image)

    return image_list


def save_tensor_images(images, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()

        # Convert the PyTorch tensor to a PIL image
        image = transforms.ToPILImage()(image)

        # Save the image with a unique filename
        image_path = os.path.join(folder_path, f"image_{i}.png")
        image.save(image_path)
