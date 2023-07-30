import torch
import torch.nn as nn
import lightning.pytorch as l
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.utils import make_grid

from skeleton.models.utils import grayscale_to_rgb


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc,
                 n_residual_blocks=6):  # TODO: somewhere they mention that n_residual_blocks should be 9
        super(Generator, self).__init__()

        # Downsample
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(n_residual_blocks)])

        # Upsample
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.in4 = nn.InstanceNorm2d(128, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.in5 = nn.InstanceNorm2d(64, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Downsample
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu3(x)

        # Residual blocks
        x = self.residual_blocks(x)

        # Upsample
        x = self.conv4(x)
        x = self.in4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.in5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_shape):
        super(Discriminator, self).__init__()

        # define architecture
        self.seq = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False),
        )

        # get the output tensor shape
        self.output_shape = self._get_conv_output_size(output_shape)

    def _get_conv_output_size(self, input_size):
        _input = torch.zeros(1, self.seq[0].in_channels, *input_size)
        output = self.forward(_input)
        return output.size()[1:]

    def forward(self, x):
        x = self.seq(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.in2(out)

        out += residual
        return out


class CycleGAN(l.LightningModule):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, lr=0.0002, beta1=0.5, lambda_idt=0.5, lambda_cycle=5,
                 image_shape=(240, 320), log_nth_image=100, scheduler_step_freq=10, scheduler_enabled=False,
                 scheduler_n_steps=500, scheduler_eta_min=2e-5):
        super(CycleGAN, self).__init__()

        # saves all arguments of __init__() as hyperparameters
        self.save_hyperparameters()

        # define Generators and Discriminators
        self.generator_g2f = Generator(input_nc, output_nc, n_residual_blocks)
        self.generator_f2g = Generator(output_nc, input_nc, n_residual_blocks)
        self.discriminator_x = Discriminator(input_nc, image_shape)
        self.discriminator_y = Discriminator(output_nc, image_shape)

        # set loss functions
        self.cycle_consistence_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # TODO BCE or MSE?
        self.discriminator_loss = nn.MSELoss()

        # set automatic optimization to false since we are using multiple optimizers for each model
        self.automatic_optimization = False

    def forward(self, x):
        return self.generator_g2f(x)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator_g2f.parameters(), lr=self.hparams.lr,
                                       betas=(self.hparams.beta1, 0.999))
        optimizer_f = torch.optim.Adam(self.generator_f2g.parameters(), lr=self.hparams.lr,
                                       betas=(self.hparams.beta1, 0.999))
        optimizer_d_x = torch.optim.Adam(self.discriminator_x.parameters(), lr=self.hparams.lr,
                                         betas=(self.hparams.beta1, 0.999))
        optimizer_d_y = torch.optim.Adam(self.discriminator_y.parameters(), lr=self.hparams.lr,
                                         betas=(self.hparams.beta1, 0.999))

        scheduler_g = CosineAnnealingWarmRestarts(optimizer_g, self.hparams.scheduler_n_steps,
                                                  eta_min=self.hparams.scheduler_eta_min, verbose=True)
        scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, self.hparams.scheduler_n_steps,
                                                  eta_min=2e-5, verbose=True)
        scheduler_d_x = CosineAnnealingWarmRestarts(optimizer_d_x, self.hparams.scheduler_n_steps,
                                                    eta_min=2e-5, verbose=True)
        scheduler_d_y = CosineAnnealingWarmRestarts(optimizer_d_y, self.hparams.scheduler_n_steps,
                                                    eta_min=2e-5, verbose=True)

        return [optimizer_g, optimizer_f, optimizer_d_x, optimizer_d_y], \
               [scheduler_g, scheduler_f, scheduler_d_x, scheduler_d_y]

    def backpropagate_loss(self, optimizer, loss, loss_name):
        optimizer.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        optimizer.step()

    def training_step(self, batch, batch_idx):
        # get source and target image
        real_x, real_y = batch

        # Generate fake images
        fake_y = self.generator_g2f(real_x)
        fake_x = self.generator_f2g(real_y)

        # Generate identities if identity loss included
        if self.hparams.lambda_idt != 0:
            id_y = self.generator_g2f(real_y)
            id_x = self.generator_f2g(real_x)
        else:
            id_y = 0
            id_x = 0

        # get optimizers and schedulers for each model
        optimizer_g, optimizer_f, optimizer_d_x, optimizer_d_y = self.optimizers()
        scheduler_g, scheduler_f, scheduler_d_x, scheduler_d_y = self.lr_schedulers()

        # ================== BACKPROPAGATE ==================
        # ---------------- TRAIN GENERATOR G ----------------

        self.toggle_optimizer(optimizer_g)

        # Generate reconstructed image
        rec_x = self.generator_f2g(fake_y)

        # calculate loss
        discrimined_y = self.discriminator_y(fake_y)
        g_adv_loss = self.adversarial_loss(discrimined_y, torch.ones_like(discrimined_y))
        g_cycle_consistency_loss = self.cycle_consistence_loss(rec_x, real_x)
        g_identity_loss = self.identity_loss(id_y, real_y) if self.hparams.lambda_idt != 0 else 0
        g_loss = g_adv_loss + self.hparams.lambda_cycle * g_cycle_consistency_loss + \
                 self.hparams.lambda_idt * g_identity_loss

        # print(f"{g_adv_loss} + {self.hparams.lambda_cycle * g_cycle_consistency_loss} + {self.hparams.lambda_idt * g_identity_loss}")

        # backpropagate the loss
        self.backpropagate_loss(optimizer_g, g_loss, "g_loss")

        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_g.step()

        self.log_dict({
            "g_adv_loss": g_adv_loss,
            "g_cycle_consistency_loss": g_cycle_consistency_loss,
            "g_identity_loss": g_identity_loss,
            "g_loss": g_loss
        })

        self.untoggle_optimizer(optimizer_g)

        # ---------------- TRAIN GENERATOR F ----------------

        self.toggle_optimizer(optimizer_f)

        # generate reconstructed image
        rec_y = self.generator_g2f(fake_x)

        # calculate loss
        discrimined_x = self.discriminator_x(fake_x)
        f_adv_loss = self.adversarial_loss(discrimined_x, torch.ones_like(discrimined_x))
        f_cycle_consistency_loss = self.cycle_consistence_loss(rec_y, real_y)
        f_identity_loss = self.identity_loss(id_x, real_x) if self.hparams.lambda_idt != 0 else 0
        f_loss = f_adv_loss + self.hparams.lambda_cycle * f_cycle_consistency_loss \
                 + self.hparams.lambda_idt * f_identity_loss

        # backpropagate the loss
        self.backpropagate_loss(optimizer_f, f_loss, "f_loss")

        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_f.step()

        self.log_dict({
            "f_adv_loss": f_adv_loss,
            "f_cycle_consistency_loss": f_cycle_consistency_loss,
            "f_identity_loss": f_identity_loss,
            "f_loss": f_loss,
        })

        self.untoggle_optimizer(optimizer_f)

        # ---------------- TRAIN DISCRIMINATOR X ------------

        self.toggle_optimizer(optimizer_d_x)

        # recalculate for pytorch fixme?
        fake_x_2 = self.generator_f2g(real_y)

        # discriminate the real and fake images
        x_dis_real = self.discriminator_x(real_x)
        x_dis_fake = self.discriminator_x(fake_x_2)

        # calculate loss
        d_x_loss_real = self.discriminator_loss(x_dis_real, torch.ones_like(x_dis_real))
        d_x_loss_fake = self.discriminator_loss(x_dis_fake, torch.zeros_like(x_dis_fake))
        d_x_loss = d_x_loss_real + d_x_loss_fake

        # backpropagate the loss
        self.backpropagate_loss(optimizer_d_x, d_x_loss, "d_x_loss")

        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_d_x.step()

        self.log_dict({
            "d_x_loss_real": d_x_loss_real,
            "d_x_loss_fake": d_x_loss_fake,
            "d_x_loss": d_x_loss
        })

        self.untoggle_optimizer(optimizer_d_x)

        # ---------------- TRAIN DISCRIMINATOR Y ------------

        self.toggle_optimizer(optimizer_d_y)

        # recalculate for pytorch fixme?
        fake_y_2 = self.generator_g2f(real_x)

        # discriminate the real and fake images
        y_dis_real = self.discriminator_y(real_y)
        y_dis_fake = self.discriminator_y(fake_y_2)

        # calculate the loss
        d_y_loss_real = self.discriminator_loss(y_dis_real, torch.ones_like(y_dis_real))
        d_y_loss_fake = self.discriminator_loss(y_dis_fake, torch.zeros_like(y_dis_fake))
        d_y_loss = d_y_loss_real + d_y_loss_fake

        # backpropagate the loss
        self.backpropagate_loss(optimizer_d_y, d_y_loss, "d_y_loss")

        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_d_y.step()

        self.log_dict({
            "d_y_loss_real": d_y_loss_real,
            "d_y_loss_fake": d_y_loss_fake,
            "d_y_loss": d_y_loss
        })

        self.untoggle_optimizer(optimizer_d_y)

        # ===================================================

        # log images
        if self.global_step % (4 * self.hparams.log_nth_image) == 0:
            grid = make_grid([real_x[0], grayscale_to_rgb(fake_y[0]), rec_x[0],
                              grayscale_to_rgb(real_y[0]), fake_x[0], grayscale_to_rgb(rec_y[0])], nrow=3)
            self.logger.experiment.add_image("generated_images", grid, self.global_step / 4)
