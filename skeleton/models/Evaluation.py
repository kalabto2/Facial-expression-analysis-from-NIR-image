import cv2
import torch
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim


class ImageEvaluator:
    def __init__(self, generated_image, target_image, split="test"):
        self.generated_image = generated_image[0].cpu()
        self.target_image = target_image[0].cpu()
        self.split = split

        # define the evaluation metrics
        self.SSIM = None
        self.PSNR = None
        self.EN = None
        self.SIFT = None
        self.color_distance = None

    def calculate_SSIM(self):
        self.SSIM = 0

        # go through all channels
        for i in range(self.target_image.size()[0]):
            self.SSIM += ssim(
                self.target_image[i].to(torch.int).numpy(),
                self.generated_image[i].to(torch.int).numpy(),
                multichannel=True,
            )

    def calculate_PSNR(self):
        mse = torch.mean((self.target_image - self.generated_image) ** 2)
        max_pixel_value = torch.max(self.target_image)
        self.PSNR = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))

    def calculate_EN(self):
        target_histogram = cv2.calcHist(
            [self.target_image.numpy()], [0], None, [256], [0, 256]
        )
        generated_histogram = cv2.calcHist(
            [self.generated_image.numpy()], [0], None, [256], [0, 256]
        )
        self.EN = entropy(target_histogram, generated_histogram)[0]

    def calculate_color_distance(self):
        self.color_distance = cv2.norm(
            self.target_image.numpy(), self.generated_image.numpy(), cv2.NORM_L2
        )

    def calculate_SIFT(self):
        sift = cv2.SIFT_create()

        keypoints_target, descriptors_target = sift.detectAndCompute(
            self.target_image.numpy().astype("uint8"), None
        )
        keypoints_generated, descriptors_generated = sift.detectAndCompute(
            self.generated_image.numpy().astype("uint8"), None
        )

        # Create a Brute Force Matcher
        bf = cv2.BFMatcher()

        # Match descriptors
        matches = bf.knnMatch(descriptors_target, descriptors_generated, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        self.SIFT = len(good_matches)

    def setup(self):
        self.calculate_SSIM()
        self.calculate_PSNR()
        self.calculate_EN()
        self.calculate_SIFT()
        self.calculate_color_distance()

    def get_eval_metrics(self):
        prefix = f"{self.split}_"
        return {
            prefix + "SSIM": self.SSIM,
            prefix + "PSNR": self.PSNR,
            prefix + "EN": self.EN,
            prefix + "SIFT": self.SIFT,
            prefix + "color_distance": self.color_distance,
        }

    def __call__(self):
        self.setup()
        return self.get_eval_metrics()
