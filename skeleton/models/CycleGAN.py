import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F


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


class CycleGAN(pl.LightningModule):
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

    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    # def train_discriminator(self, discriminator, real, fake):
    #     pred_real = discriminator(real)
    #     loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
    #     pred_fake = discriminator(fake.detach())
    #     loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
    #     loss_discriminator = (loss_real + loss_fake) * 0.5
    #     return loss_discriminator

    def adversarial_loss(self, x, y):
        x_l = self.discriminator_x(self.generator_f2g(y))
        y_l = self.discriminator_y(self.generator_g2f(x))
        val_x = self.generator_loss(x_l, torch.ones_like(x_l))
        val_y = self.generator_loss(y_l, torch.ones_like(y_l))
        return (val_x + val_y) / 2

    def cycle_consistency_loss(self, x, y):
        reconstruction_x = self.mae(self.generator_f2g(self.generator_g2f(x)), x)
        reconstruction_y = self.mae(self.generator_g2f(self.generator_f2g(y)), y)
        return (reconstruction_x + reconstruction_y) / 2

    def identity_loss(self, x, y):
        id_x = self.mae(self.generator_f2g(x), x)
        id_y = self.mae(self.generator_g2f(y), y)
        return (id_x + id_y) / 2

    def discriminator_loss_computation(self, x_real, y_real):
        d = self.discriminator_x(self.generator_f2g(y_real))
        d_x_gen_loss = self.discriminator_loss(d, torch.zeros_like(d))
        d_y_gen_loss = self.discriminator_loss(self.discriminator_y(self.generator_g2f(x_real)),
                                               torch.zeros_like(d))
        d_x_valid_loss = self.discriminator_loss(self.discriminator_x(x_real), torch.ones_like(d))
        d_y_valid_loss = self.discriminator_loss(self.discriminator_y(y_real), torch.ones_like(d))

        d_gen_loss = (d_x_gen_loss + d_y_gen_loss) / 2

        d_loss = (d_gen_loss + d_x_valid_loss + d_y_valid_loss) / 3

        return d_loss

    def backpropagate_loss(self, optimizer, loss, loss_name):
        self.log(loss_name, loss, prog_bar=True)
        self.manual_backward(loss, retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    def generator_loss_computation(self, real_x, real_y):
        adv_loss = self.adversarial_loss(real_x, real_y)
        cycle_cons_loss = self.cycle_consistency_loss(real_x, real_y)
        id_loss = self.identity_loss(real_x, real_y)
        return adv_loss + self.lambda_cycle * cycle_cons_loss + self.lambda_idt * id_loss

    def training_step(self, batch, batch_idx):
        # get source and target image
        real_x, real_y = batch

        # # Generate fake images
        # fake_y = self.generator_g2f(real_x)
        # fake_x = self.generator_f2g(real_y)

        # # Generate identities
        # id_y = self.generator_g2f(real_y)
        # id_x = self.generator_f2g(real_x)
        #
        # # Generate reconstructed image
        # rec_y = self.generator_g2f(fake_x)
        # rec_x = self.generator_f2g(fake_y)

        # get optimizers for each model
        optimizer_g, optimizer_f, optimizer_d_x, optimizer_d_y = self.optimizers()

        # calculate generators loss
        generator_loss = self.generator_loss_computation(real_x, real_y)

        # calculate discriminators loss
        discriminator_loss = self.discriminator_loss_computation(real_x, real_y)

        # ================== BACKPROPAGATE ==================
        # ---------------- TRAIN GENERATOR G ----------------

        # TODO? Log the image?

        self.toggle_optimizer(optimizer_g)


        # # calculate losses
        # adv_loss = self.adversarial_loss(real_x, real_y)
        # cycle_cons_loss = self.cycle_consistency_loss(real_x, real_y)
        # id_loss = self.identity_loss(real_x, real_y)
        # g_loss = adv_loss + self.lambda_cycle * cycle_cons_loss + self.lambda_idt * id_loss

        # g_loss = self.generator_loss_computation(real_x, real_y)

        # # Backpropagate the loss
        # self.log("g_loss", g_loss, prog_bar=True)
        # self.manual_backward(g_loss)
        # optimizer_g.step()
        # optimizer_g.zero_grad()

        self.backpropagate_loss(optimizer_g, generator_loss, "g_loss")

        self.untoggle_optimizer(optimizer_g)

        # ---------------- TRAIN GENERATOR F ----------------

        # TODO? Log the image?

        self.toggle_optimizer(optimizer_f)

        # # calculate losses
        # adv_loss = self.adversarial_loss(real_x, real_y)
        # cycle_cons_loss = self.cycle_consistency_loss(real_x, real_y)
        # id_loss = self.identity_loss(real_x, real_y)
        # f_loss = adv_loss + self.lambda_cycle * cycle_cons_loss + self.lambda_idt * id_loss

        f_loss = self.generator_loss_computation(real_x, real_y)

        # # Backpropagate the loss
        # self.log("f_loss", f_loss, prog_bar=True)
        # self.manual_backward(f_loss)
        # optimizer_f.step()
        # optimizer_f.zero_grad()

        self.backpropagate_loss(optimizer_f, f_loss, "f_loss")

        self.untoggle_optimizer(optimizer_f)

        # ---------------- TRAIN DISCRIMINATOR X ------------

        self.toggle_optimizer(optimizer_d_x)

        d_x_loss = self.discriminator_loss_computation(real_x, real_y)

        self.backpropagate_loss(optimizer_d_x, d_x_loss, "d_x_loss")

        self.untoggle_optimizer(optimizer_d_x)

        # ---------------- TRAIN DISCRIMINATOR Y ------------

        self.toggle_optimizer(optimizer_d_y)

        d_y_loss = self.discriminator_loss_computation(real_x, real_y)

        self.backpropagate_loss(optimizer_d_y, d_y_loss, "d_y_loss")

        self.untoggle_optimizer(optimizer_d_y)

        # # Adversarial ground truths
        # valid = torch.ones((real_x.size(0), *self.discriminator_x.output_shape))
        # fake = torch.zeros((real_x.size(0), *self.discriminator_x.output_shape))
        #
        # # Generate fake images
        # fake_y = self.generator_g(real_x)
        # fake_x = self.generator_f(real_y)
        #
        # # Train generators
        # if optimizer_idx == 0:
        #     # Identity loss
        #     identity_x = self.generator_f(real_x)
        #     identity_y = self.generator_g(real_y)
        #     loss_identity = self.criterion_identity(identity_x, real_x) + self.criterion_identity(identity_y, real_y)
        #
        #     # GAN loss
        #     loss_gan_g = self.criterion_GAN(self.discriminator_y(fake_y), valid)
        #     loss_gan_f = self.criterion_GAN(self.discriminator_x(fake_x), valid)
        #     loss_generator = loss_gan_g + loss_gan_f + self.lambda_idt * loss_identity
        #
        #     self.log('train_loss_generator', loss_generator)
        #     return loss_generator
        #
        # # Train discriminators
        # elif optimizer_idx == 1:
        #     loss_discriminator_x = self.train_discriminator(self.discriminator_x, real_x, fake_x)
        #     self.log('train_loss_discriminator_x', loss_discriminator_x)
        #     return loss_discriminator_x
        #
        # elif optimizer_idx == 2:
        #     loss_discriminator_y = self.train_discriminator(self.discriminator_y, real_y, fake_y)
        #     self.log('train_loss_discriminator_y', loss_discriminator_y)
        #     return loss_discriminator_y

    # def on_train_epoch_end(self, outputs):
    #     # Log images
    #     if self.current_epoch % self.sample_interval == 0:
    #         self.sample_images()

    # def configure_optimizers(self):
    #     # Optimizers
    #     optimizer_g = torch.optim.Adam(
    #         itertools.chain(self.generator_g.parameters(), self.generator_f.parameters()),
    #         lr=self.lr, betas=(self.beta1, 0.999))
    #
    #     optimizer_d_x = torch.optim.Adam(self.discriminator_x.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
    #     optimizer_d_y = torch.optim.Adam(self.discriminator_y.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
    #
    #     return {
    #         'optimizer': [optimizer_g, optimizer_d_x, optimizer_d_y],
    #         'frequency': self.discriminator_freq,
    #         'interval': self.generator_freq
    #     }

    # def sample_images(self):
    #     """Saves a generated sample from the validation set"""
    #     real_x, real_y = next(iter(self.val_dataloader()))
    #     real_x = real_x.to(self.device)
    #     real_y = real_y.to(self.device)
    #     fake_y = self.generator_g(real_x)
    #     fake_x = self.generator_f(real_y)
    #     img_sample = torch.cat((real_x.data, fake_y.data, real_y.data, fake_x.data), -2)
    #     save_image(img_sample, f"{self.logger.log_dir}/epoch-{self.current_epoch}.png", nrow=8, normalize=True)
