import torch


def grayscale_to_rgb(grayscale_image):
    """Converts a tensor grayscale image to an RGB image."""
    rgb_image = torch.zeros((3, grayscale_image.shape[1], grayscale_image.shape[2]))
    for i in range(3):
        rgb_image[i] = grayscale_image

    if grayscale_image.is_cuda:
        rgb_image = rgb_image.cuda()

    return rgb_image
