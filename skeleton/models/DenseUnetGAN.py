import math

import torch
import torch.nn as nn
import lightning.pytorch as l
from torchvision.transforms import GaussianBlur
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from skeleton.models.Evaluation import ImageEvaluator
from skeleton.models.networks import DenseUnet, PatchDiscriminatorPaired
from skeleton.models.utils import grayscale_to_rgb


class DenseUnetGAN(l.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_shape,
        lr,
        beta1,
        weights_init_std,
        l_disc=1,
        l_color=0.0004,
        l_pix=40,
        l_feature=1.3,
        log_nth_image=100,
        device="cuda",
        d_every_n_step=2,
    ):
        super(DenseUnetGAN, self).__init__()

        # saves all arguments of __init__() as hyperparameters
        self.save_hyperparameters()

        # discriminator and generator
        self.disc = PatchDiscriminatorPaired()
        self.gen = DenseUnet(in_channels, out_channels)

        # init weights with normal distribution
        self.weights_init(self.disc, self.hparams.weights_init_std)
        self.weights_init(self.gen, self.hparams.weights_init_std)

        # set automatic optimization to false since we are using multiple optimizers for each model
        self.automatic_optimization = False

        # set loss functions
        self.pix_loss = nn.L1Loss()
        self.color_loss = nn.MSELoss()
        # TODO BCE or MSE? - in paper they use MSE in reality, tho show BCE
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.discriminator_loss = nn.MSELoss()

        # prepare VGG19 for feature loss
        vgg19_model = models.vgg19(pretrained=True)
        if device == "gpu":
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
        self.vgg19 = lambda x: vgg19_model(transform(x[0]).unsqueeze(0))

        self.image_evaluator = ImageEvaluator(device=device)

    def forward(self, x):
        return self.gen(x[0].unsqueeze(0).unsqueeze(0))

    def calculate_loss(self, fake_y, y, d_real, d_fake):
        # prepare Gaussian kernel for color loss
        gaussian_blur = GaussianBlur((15, 15), math.sqrt(3))

        # calculate generator losses
        color_loss = self.hparams.l_color * self.color_loss(
            gaussian_blur(fake_y), gaussian_blur(y)
        )
        pix_loss = self.hparams.l_pix * self.pix_loss(fake_y, y)
        feature_loss = self.hparams.l_feature * torch.norm(
            self.vgg19(y) - self.vgg19(fake_y), p=2
        )
        adversarial_loss = self.adversarial_loss(d_fake, torch.ones_like(d_fake))
        gan_loss = adversarial_loss + color_loss + pix_loss + feature_loss

        # calculate discriminator loss
        d_loss_real = self.discriminator_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = self.discriminator_loss(d_fake, torch.zeros_like(d_fake))
        disc_loss = self.hparams.l_disc * (d_loss_real + d_loss_fake)

        # log
        self.log_dict(
            {
                "gen_col_loss": color_loss,
                "gen_pix_loss": pix_loss,
                "gen_feat_loss": feature_loss,
                "gen_adv_loss": adversarial_loss,
                "gen_loss": gan_loss,
                "disc_loss": disc_loss,
            }
        )

        return gan_loss, disc_loss

    @staticmethod
    def weights_init(model, std=0.02):
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            elif param.dim() >= 2:
                if name.startswith(
                    "layers.0"
                ):  # The first layer does not have ReLU applied on its input
                    param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
                else:
                    param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))
            else:
                param.data.normal_(mean=0.0, std=std)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.gen.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999)
        )
        opt_d = torch.optim.Adam(
            self.disc.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
        )

        return [opt_g, opt_d], []

    def log_image(self, img_arr, nrows, split):
        grid = make_grid(
            img_arr,
            nrow=nrows,
        )
        self.logger.experiment.add_image(
            f"{split}_generated_images", grid, self.global_step / 2
        )

    def training_step(self, batch, batch_idx):
        y, x = batch

        # ======== BACKPROP GEN ===============

        fake_y = self.gen(x)
        # d_real = self.disc((x.clone(), y.clone()))
        d_fake = self.disc((x, fake_y))

        # get optimizers and schedulers for each model
        opt_g, opt_d = self.optimizers()

        # prepare Gaussian kernel for color loss
        gaussian_blur = GaussianBlur((15, 15), math.sqrt(3))

        # calculate generator losses
        color_loss = self.hparams.l_color * self.color_loss(
            gaussian_blur(fake_y), gaussian_blur(y)
        )
        pix_loss = self.hparams.l_pix * self.pix_loss(fake_y, y)
        feature_loss = self.hparams.l_feature * torch.norm(
            self.vgg19(y) - self.vgg19(fake_y), p=2
        )
        adversarial_loss = self.adversarial_loss(d_fake, torch.ones_like(d_fake))
        gan_loss = adversarial_loss + color_loss + pix_loss + feature_loss

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        self.manual_backward(gan_loss, retain_graph=True)
        self.untoggle_optimizer(opt_g)

        opt_g.step()

        # =========== BACKPROP GEN AGAIN ======

        fake_y_2 = self.gen(x)
        # d_real = self.disc((x.clone(), y.clone()))
        d_fake_2 = self.disc((x, fake_y_2))

        # calculate generator losses
        color_loss_2 = self.hparams.l_color * self.color_loss(
            gaussian_blur(fake_y_2), gaussian_blur(y)
        )
        pix_loss_2 = self.hparams.l_pix * self.pix_loss(fake_y_2, y)
        feature_loss_2 = self.hparams.l_feature * torch.norm(
            self.vgg19(y) - self.vgg19(fake_y_2), p=2
        )
        adversarial_loss_2 = self.adversarial_loss(d_fake_2, torch.ones_like(d_fake_2))
        gan_loss_2 = adversarial_loss_2 + color_loss_2 + pix_loss_2 + feature_loss_2

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        self.manual_backward(gan_loss_2, retain_graph=True)
        self.untoggle_optimizer(opt_g)

        opt_g.step()

        # =========== BACKPROP DISC ===========

        fake_y_3 = self.gen(x)
        d_real_2 = self.disc((x, y))
        d_fake_2 = self.disc((x, fake_y_3))

        # calculate discriminator loss
        d_loss_real = self.discriminator_loss(d_real_2, torch.ones_like(d_real_2))
        d_loss_fake = self.discriminator_loss(d_fake_2, torch.zeros_like(d_fake_2))
        disc_loss = self.hparams.l_disc * (d_loss_real + d_loss_fake)

        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        self.manual_backward(disc_loss, retain_graph=True)
        self.untoggle_optimizer(opt_d)

        opt_d.step()

        # =====================================

        # log images
        # if self.global_step % (2 * self.hparams.log_nth_image) == 0:
        grid = make_grid(
            [
                grayscale_to_rgb(x[0]),
                y[0],
                fake_y[0],
            ],
            nrow=3,
        )
        self.logger.experiment.add_image("generated_images", grid, self.global_step / 2)

        # log
        self.log_dict(
            {
                "gen_col_loss": color_loss_2,
                "gen_pix_loss": pix_loss_2,
                "gen_feat_loss": feature_loss_2,
                "gen_adv_loss": adversarial_loss_2,
                "gen_loss": gan_loss_2,
                "disc_loss": disc_loss,
            }
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        y = batch[0][0].unsqueeze(0)
        x = batch[0][1][0].unsqueeze(0).unsqueeze(0)

        fake_y = self.gen(x)
        d_real = self.disc((x, y))
        d_fake = self.disc((x, fake_y))

        eval_metrics_y = self.image_evaluator(fake_y, y, "test-y")

        self.log_dict(eval_metrics_y)

        # log one image
        self.log_image(
            [grayscale_to_rgb(x[0]), y[0], fake_y[0]],
            3,
            "test",
        )

        # calculate losses for discriminator
        d_loss_real = self.discriminator_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = self.discriminator_loss(d_fake, torch.zeros_like(d_fake))
        disc_loss = self.hparams.l_disc * (d_loss_real + d_loss_fake)

        self.log_dict(
            {
                "test_disc_loss": disc_loss,
                "test_disc_real_loss": d_loss_real,
                "test_disc_fake_loss": d_loss_fake,
            }
        )

    def validation_step(self, batch, batch_idx):
        y = batch[0][0].unsqueeze(0)
        x = batch[0][1][0].unsqueeze(0).unsqueeze(0)

        fake_y = self.gen(x)
        d_real = self.disc((x, y))
        d_fake = self.disc((x, fake_y))

        eval_metrics_y = self.image_evaluator(fake_y, y, "val-y")

        self.log_dict(eval_metrics_y)

        # log one image
        self.log_image(
            [grayscale_to_rgb(x[0]), y[0], fake_y[0]],
            3,
            "val",
        )

        # calculate losses for discriminator
        d_loss_real = self.discriminator_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = self.discriminator_loss(d_fake, torch.zeros_like(d_fake))
        disc_loss = self.hparams.l_disc * (d_loss_real + d_loss_fake)

        self.log_dict(
            {
                "val_disc_loss": disc_loss,
                "val_disc_real_loss": d_loss_real,
                "val_disc_fake_loss": d_loss_fake,
            }
        )
