import torch
import torch.nn as nn
import lightning.pytorch as l
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LambdaLR,
    CosineAnnealingLR,
)
from torchvision.utils import make_grid

from skeleton.models.Evaluation import ImageEvaluator
from skeleton.models.utils import grayscale_to_rgb
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6):
        super(Generator, self).__init__()

        # Downsample
        self.conv1 = nn.Conv2d(
            input_nc,
            64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            padding_mode="reflect",
        )
        # self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        self.relu3 = nn.ReLU(inplace=False)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(n_residual_blocks)]
        )

        # Upsample
        self.conv4 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.in4 = nn.InstanceNorm2d(128, affine=True)
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.in5 = nn.InstanceNorm2d(64, affine=True)
        self.relu5 = nn.ReLU(inplace=False)
        self.conv6 = nn.Conv2d(
            64, output_nc, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Downsample
        x = self.conv1(x)
        # x = self.in1(x)
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
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
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


class ResBlockConcat(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockConcat, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_skip = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        skip = self.conv_skip(x)
        skip = self.bn_skip(skip)

        out = torch.cat((out, skip), dim=1)
        out = F.relu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.in2(out)

        # out += residual
        out = out + residual
        return out


class CycleGAN(l.LightningModule):
    def __init__(
        self,
        input_nc,
        output_nc,
        n_residual_blocks=6,
        lr=0.0002,
        beta1=0.5,
        lambda_idt=0.5,
        lambda_cycle=5,
        image_shape=(240, 320),
        log_nth_image=100,
        scheduler_step_freq=10,
        scheduler_enabled=False,
        scheduler_n_steps=500,
        scheduler_eta_min=2e-5,
        weights_init_std=0.02,
        lambda_discriminator=0.5,
        scheduler="linear",
        linear_lr_w_init_lr=5,
        linear_lr_w_decay=5,
    ):
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

        # initialize weights
        self.weights_init(self.generator_g2f, std=self.hparams.weights_init_std)
        self.weights_init(self.generator_f2g, std=self.hparams.weights_init_std)
        self.weights_init(self.discriminator_x, std=self.hparams.weights_init_std)
        self.weights_init(self.discriminator_y, std=self.hparams.weights_init_std)
        # visualize_activations(model, print_variance=True) # TODO add visualization?

    def forward(self, x):
        return self.generator_g2f(x)

    def weights_init(self, model, std=0.02):
        for name, param in model.named_parameters():
            param.data.normal_(mean=0.0, std=std)

    # def init_weights(self, net, init_type='normal', std=0.02):
    #     def init_func(m):  # define the initialization function
    #         classname = m.__class__.__name__
    #         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
    #             if init_type == 'normal':
    #                 init.normal_(m.weight.data, 0.0, std)
    #             elif init_type == 'xavier':
    #                 init.xavier_normal_(m.weight.data, gain=std)
    #             elif init_type == 'kaiming':
    #                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 init.orthogonal_(m.weight.data, gain=std)
    #             else:
    #                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)
    #         elif classname.find(
    #                 'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    #             init.normal_(m.weight.data, 1.0, std)
    #             init.constant_(m.bias.data, 0.0)
    #
    #     print('initialize network with %s' % init_type)
    #     net.apply(init_func)  # apply the initialization function <init_func>

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(
            self.generator_g2f.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
        )
        optimizer_f = torch.optim.Adam(
            self.generator_f2g.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
        )
        optimizer_d_x = torch.optim.Adam(
            self.discriminator_x.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
        )
        optimizer_d_y = torch.optim.Adam(
            self.discriminator_y.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
        )
        optimizers = [optimizer_g, optimizer_f, optimizer_d_x, optimizer_d_y]

        if self.hparams.scheduler == "linear":

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - self.hparams.linear_lr_w_init_lr) / float(
                    self.hparams.linear_lr_w_decay + 1
                )
                return lr_l

            schedulers = [
                LambdaLR(optimizer, lr_lambda=lambda_rule) for optimizer in optimizers
            ]
        elif self.hparams.scheduler == "cos_ann_w_rest":
            schedulers = [
                CosineAnnealingWarmRestarts(
                    optimizer,
                    self.hparams.scheduler_n_steps,
                    eta_min=self.hparams.scheduler_eta_min,
                    verbose=True,
                )
                for optimizer in optimizers
            ]
        elif self.hparams.scheduler == "cos_ann":
            raise Exception("Not implemented")
            # schedulers = [CosineAnnealingLR(optimizer, self.hparams.scheduler_n_steps,
            #               eta_min=self.hparams.scheduler_eta_min, verbose=True) for optimizer in optimizers]
        else:
            schedulers = []

        return optimizers, schedulers

    def calculate_loss_generator(self, discr, total_cycle_consistency_loss, id, real, gen_id):
        adv_loss = self.adversarial_loss(discr, torch.ones_like(discr))
        id_loss = (
                self.identity_loss(id, real)
                if self.hparams.lambda_idt != 0
                else torch.tensor(0, dtype=torch.float32)
            )
        total_loss = (
                adv_loss
                + self.hparams.lambda_cycle * total_cycle_consistency_loss
                + self.hparams.lambda_idt * id_loss
            )

        # log losses
        self.log_dict(
            {
                f"{gen_id}_adv_loss": adv_loss,
                f"{gen_id}_cycle_consistency_loss": total_cycle_consistency_loss,
                f"{gen_id}_identity_loss": id_loss,
                f"{gen_id}_loss": total_loss,
            }
        )

        return total_loss

    def calculate_loss_discriminator(self, d_real, d_fake, d_id):
        # calculate loss
        d_loss_real = self.discriminator_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = self.discriminator_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = self.hparams.lambda_discriminator * (d_loss_real + d_loss_fake)

        # log losses
        self.log_dict(
            {
                f"d_{d_id}_loss_real": d_loss_real,
                f"d_{d_id}_loss_fake": d_loss_fake,
                f"d_{d_id}_loss": d_loss,
            }
        )

        return d_loss

    def generate(self, batch):
        # get source and target image
        self.real_x, self.real_y = batch

        # Generate fake images
        self.fake_y = self.generator_g2f(self.real_x)
        self.fake_x = self.generator_f2g(self.real_y)

        # Generate reconstructed image
        self.rec_x = self.generator_f2g(self.fake_y)
        self.rec_y = self.generator_g2f(self.fake_x)

        # Generate identities if identity loss included
        self.id_y = self.generator_g2f(self.real_y) if self.hparams.lambda_idt != 0 else 0
        self.id_x = self.generator_f2g(self.real_x) if self.hparams.lambda_idt != 0 else 0

        # Discriminate fake generated images
        self.discrimined_y = self.discriminator_y(self.fake_y)
        self.discrimined_x = self.discriminator_x(self.fake_x)

        # Discriminate real generated images
        self.discrimined_y_real = self.discriminator_y(self.real_y)
        self.discrimined_x_real = self.discriminator_x(self.real_x)

    def calculate_losses(self):
        total_cycle_consistency_loss = (self.cycle_consistence_loss(self.rec_x, self.real_x) +
                                        self.cycle_consistence_loss(self.rec_y, self.real_y))

        self.g_loss = self.calculate_loss_generator(self.discrimined_y, total_cycle_consistency_loss, self.id_y, self.real_y, "g")
        self.f_loss = self.calculate_loss_generator(self.discrimined_x, total_cycle_consistency_loss, self.id_x, self.real_x, "f")
        self.d_x_loss = self.calculate_loss_discriminator(self.discrimined_x_real, self.discrimined_x, "x")
        self.d_y_loss = self.calculate_loss_discriminator(self.discrimined_y_real, self.discrimined_y, "y")

    def training_step(self, batch, batch_idx):
        # generate all images, discriminators
        self.generate(batch)

        # calculate losses
        self.calculate_losses()

        # get optimizers and schedulers for each model
        optimizer_g, optimizer_f, optimizer_d_x, optimizer_d_y = self.optimizers()
        if self.hparams.scheduler_enabled:
            (
                scheduler_g,
                scheduler_f,
                scheduler_d_x,
                scheduler_d_y,
            ) = self.lr_schedulers()

        # ================== BACKPROPAGATE ==================
        # ---------------- TRAIN GENERATOR G ----------------
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        self.manual_backward(self.g_loss, retain_graph=True)
        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_g.step()

        self.untoggle_optimizer(optimizer_g)

        # ---------------- TRAIN GENERATOR F ----------------
        self.toggle_optimizer(optimizer_f)
        optimizer_f.zero_grad()
        self.manual_backward(self.f_loss, retain_graph=True)
        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_f.step()
        self.untoggle_optimizer(optimizer_f)

        # ---------------- TRAIN OPTIMIZER DX ----------------
        self.toggle_optimizer(optimizer_d_x)
        optimizer_d_x.zero_grad()
        self.manual_backward(self.d_x_loss, retain_graph=False)
        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_d_x.step()

        self.untoggle_optimizer(optimizer_d_x)

        # ---------------- TRAIN OPTIMIZER DY ----------------
        self.toggle_optimizer(optimizer_d_y)
        optimizer_d_y.zero_grad()
        self.manual_backward(self.d_y_loss, retain_graph=False)
        if self.hparams.scheduler_enabled and batch_idx % self.hparams.scheduler_step_freq == 0:
            scheduler_d_y.step()

        self.untoggle_optimizer(optimizer_d_y)
        # ===================================================

        # update weights
        optimizer_f.step()
        optimizer_g.step()
        optimizer_d_x.step()
        optimizer_d_y.step()
        # ===================================================
        
        # log images
        if self.global_step % (4 * self.hparams.log_nth_image) == 0:
            grid = make_grid(
                [
                    self.real_x[0],
                    grayscale_to_rgb(self.fake_y[0]),
                    self.rec_x[0],
                    grayscale_to_rgb(self.real_y[0]),
                    self.fake_x[0],
                    grayscale_to_rgb(self.rec_y[0]),
                ],
                nrow=3,
            )
            self.logger.experiment.add_image("generated_images", grid, self.global_step / 4)

    def test_step(self, batch, batch_idx):
        x, y = batch

        # generate output
        out = self.forward(x)

        # calculate evaluation metrics
        image_evaluator = ImageEvaluator(out, y)
        eval_metrics = image_evaluator(test_prefix=True)

        # log evaluation metrics
        self.log_dict(eval_metrics)

        # log images
        if self.global_step % 100 == 0:
            grid = make_grid([grayscale_to_rgb(x[0]), y[0], out[0]], nrow=3)
            self.logger.experiment.add_image(
                "test_generated_images", grid, self.global_step / 4
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            gen = self.generator_g2f
            dis_a = self.discriminator_y
            dis_b = self.discriminator_x
            dis_id = "Y"
        else:
            gen = self.generator_f2g
            dis_a = self.discriminator_x
            dis_b = self.discriminator_y
            dis_id = "X"

        real = batch
        fake = gen(real)
        dis_real = dis_b(real)
        dis_fake = dis_a(fake)

        # calculate the loss
        d_loss_real = self.discriminator_loss(dis_real, torch.ones_like(dis_real))
        d_loss_fake = self.discriminator_loss(dis_fake, torch.zeros_like(dis_fake))
        d_loss = self.hparams.lambda_discriminator * (d_loss_real + d_loss_fake)

        # ------------------- LOG ------------------------------
        self.log_dict(
            {
                f"val_loss_D{dis_id}": d_loss,
                f"val_loss_D{dis_id}_real": d_loss_real,
                f"val_loss_D{dis_id}_fake": d_loss_fake,
            }
        )
