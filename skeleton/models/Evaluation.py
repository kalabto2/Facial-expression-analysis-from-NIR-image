import cv2
import torch
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import torchvision.models as models
import torchvision.transforms as transforms


class ImageEvaluator:
    def __init__(self, device="cpu"):
        self.generated_image = None
        self.target_image = None
        self.split = None
        self.device = device

        # define the evaluation metrics
        self.SSIM = None
        self.PSNR = None
        self.EN = None
        self.SIFT = None
        self.color_distance = None
        self.feature_loss = None

        # prepare VGG19 for feature loss
        vgg19_model = models.vgg19(pretrained=True)
        if self.device == "gpu":
            vgg19_model = vgg19_model.cuda()

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize to VGG-16 input size
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Mean values for ImageNet data
                    std=[0.229, 0.224, 0.225],  # Standard deviations for ImageNet data
                ),
            ]
        )
        self.vgg19 = lambda x: vgg19_model(transform(x))

    def calculate_feature_loss(self):
        if self.target_image.size()[0] == 1:
            target_adjusted = torch.stack([self.target_image] * 3, dim=1)
            generated_adjusted = torch.stack([self.generated_image] * 3, dim=1)
        else:
            target_adjusted = torch.unsqueeze(self.target_image, 0)
            generated_adjusted = torch.unsqueeze(self.generated_image, 0)

        self.feature_loss = torch.norm(
            self.vgg19(target_adjusted) - self.vgg19(generated_adjusted),
            p=2,
        )

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

    def setup(self, generated_image, target_image, split):
        self.split = split
        self.generated_image = generated_image[0].cpu()
        self.target_image = target_image[0].cpu()

        self.calculate_SSIM()
        self.calculate_PSNR()
        self.calculate_EN()
        self.calculate_SIFT()
        self.calculate_color_distance()

        self.generated_image = generated_image[0]
        self.target_image = target_image[0]

        self.calculate_feature_loss()

    def get_eval_metrics(self):
        prefix = f"{self.split}_"
        return {
            prefix + "SSIM": self.SSIM,
            prefix + "PSNR": self.PSNR,
            prefix + "EN": self.EN,
            prefix + "SIFT": self.SIFT,
            prefix + "color_distance": self.color_distance,
            prefix + "feature_loss": self.feature_loss,
        }

    def __call__(self, generated_image, target_image, split):
        self.setup(generated_image, target_image, split)
        return self.get_eval_metrics()
