import copy

import torch
import torch.nn as nn
import lightning.pytorch as l
import torch.nn.functional as F
from torchvision.utils import make_grid


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6):   # TODO: somewhere they menttion that n_residual_blocks should be 9
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
    def __init__(self, input_nc):
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
        self.output_shape = self._get_conv_output_size((240, 320))  # TODO refactor so it is not hardcoded

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
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, lr=0.0002, beta1=0.5, lambda_idt=0.5):
        super(CycleGAN, self).__init__()

        # TODO all those parameters to self.save_hyperparameters() -> better, bcs shorter
        self.generator_g2f = Generator(input_nc, output_nc, n_residual_blocks)
        self.generator_f2g = Generator(output_nc, input_nc, n_residual_blocks)
        self.discriminator_x = Discriminator(input_nc)
        self.discriminator_y = Discriminator(output_nc)

        self.lr = lr
        self.beta1 = beta1
        self.lambda_idt = lambda_idt

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # set automatic optimization to false since we are using multiple optimizers for each model
        self.automatic_optimization = False

        # TODO add them as arguments, not hardcoded
        self.lambda_cycle = 5

        # set losses TODO check it
        self.mae = nn.L1Loss()
        self.generator_loss = F.binary_cross_entropy_with_logits  # TODO or MSE? + add to init
        self.discriminator_loss = nn.MSELoss()

    def forward(self, x):
        return self.generator_g2f(x)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator_g2f.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizer_f = torch.optim.Adam(self.generator_f2g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizer_d_x = torch.optim.Adam(self.discriminator_x.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizer_d_y = torch.optim.Adam(self.discriminator_y.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        return [optimizer_g, optimizer_f, optimizer_d_x, optimizer_d_y], []  # TODO add here a scheduler in the 2nd list

    # def set_requires_grad(self, nets, requires_grad=False):
    #     for net in nets:
    #         for param in net.parameters():
    #             param.requires_grad = requires_grad

    def cycle_consistency_loss(self, x, y):
        reconstruction_x = self.mae(self.generator_f2g(self.generator_g2f(x)), x)
        reconstruction_y = self.mae(self.generator_g2f(self.generator_f2g(y)), y)
        return (reconstruction_x + reconstruction_y) / 2

    def identity_loss(self, x, y):
        id_x = self.mae(self.generator_f2g(x), x)
        id_y = self.mae(self.generator_g2f(y), y)
        return (id_x + id_y) / 2

    def backpropagate_loss(self, optimizer, loss, loss_name):
        self.log(loss_name, loss, prog_bar=True)
        self.manual_backward(loss, retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        # get source and target image
        real_x, real_y = batch

        # Generate fake images
        fake_y = self.generator_g2f(real_x)
        fake_x = self.generator_f2g(real_y)

        # Generate identities
        id_y = self.generator_g2f(real_y)
        id_x = self.generator_f2g(real_x)

        # Generate reconstructed image
        rec_y = self.generator_g2f(fake_x)
        rec_x = self.generator_f2g(fake_y)

        # log images - real_x, fake_y, real_y and fake_x
        grid = make_grid([real_x[0], fake_y[0], real_y[0], fake_x[0]], nrow=2)
        self.logger.experiment.add_image("generated_images", grid, self.global_step / 4)

        # get optimizers for each model
        optimizer_g, optimizer_f, optimizer_d_x, optimizer_d_y = self.optimizers()

        # ================== BACKPROPAGATE ==================
        # ---------------- TRAIN GENERATOR G ----------------

        self.toggle_optimizer(optimizer_g)

        # calculate loss
        discrimined_y = self.discriminator_y(fake_y)
        g_adv_loss = self.generator_loss(discrimined_y, torch.ones_like(discrimined_y))
        g_cycle_consistency_loss = self.mae(rec_x, real_x)
        g_identity_loss = self.mae(id_y, real_y)
        g_loss = g_adv_loss + self.lambda_cycle * g_cycle_consistency_loss + self.lambda_idt * g_identity_loss

        # backpropagate the loss
        self.backpropagate_loss(optimizer_g, g_loss, "g_loss")

        self.untoggle_optimizer(optimizer_g)

        # ---------------- TRAIN GENERATOR F ----------------

        self.toggle_optimizer(optimizer_f)

        # recalculate for pytorch fixme?
        rec_y_2 = self.generator_g2f(fake_x)

        # calculate loss
        discrimined_x = self.discriminator_x(fake_x)
        f_adv_loss = self.generator_loss(discrimined_x, torch.ones_like(discrimined_x))
        f_cycle_consistency_loss = self.mae(rec_y_2, real_y)
        f_identity_loss = self.mae(id_x, real_x)
        f_loss = f_adv_loss + self.lambda_cycle * f_cycle_consistency_loss + self.lambda_idt * f_identity_loss

        # backpropagate the loss
        self.backpropagate_loss(optimizer_f, f_loss, "f_loss")

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

        self.untoggle_optimizer(optimizer_d_y)

        # ===================================================

        # TODO add some self.log_dict()?
