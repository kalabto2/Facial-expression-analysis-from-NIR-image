import math
import pathlib

import torch
import torch.nn as nn
import lightning.pytorch as l
import torchvision
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LambdaLR,
    CosineAnnealingLR,
)
from torchvision.utils import make_grid

from skeleton.models.Evaluation import ImageEvaluator
from skeleton.models.networks import PatchDiscriminator, Generator
from skeleton.models.utils import grayscale_to_rgb, ImagePool


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
        weights_init="normal",
        weights_init_std=0.02,
        lambda_discriminator=0.5,
        scheduler="linear",
        linear_lr_w_init_lr=5,
        linear_lr_w_decay=5,
        device="cpu",
    ):
        super(CycleGAN, self).__init__()

        # saves all arguments of __init__() as hyperparameters
        self.save_hyperparameters()

        # define Generators and Discriminators
        self.generator_g2f = Generator(input_nc, output_nc, n_residual_blocks)
        self.generator_f2g = Generator(output_nc, input_nc, n_residual_blocks)
        self.discriminator_x = PatchDiscriminator(input_nc, image_shape)
        self.discriminator_y = PatchDiscriminator(output_nc, image_shape)

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

        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()

        self.image_evaluator = ImageEvaluator(device=device)

    def forward(self, x):
        return self.generator_g2f(x)

    def weights_init(self, model, std=0.02):
        if self.hparams.weights_init == "normal":
            for name, param in model.named_parameters():
                param.data.normal_(mean=0.0, std=std)
        elif self.hparams.weights_init == "xavier":
            for name, param in model.named_parameters():
                if name.endswith(".bias"):
                    param.data.fill_(0)
                elif param.dim() >= 2:
                    bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                    param.data.uniform_(-bound, bound)
                else:
                    param.data.normal_(mean=0.0, std=std)
        elif self.hparams.weights_init == "kaiming":
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
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented"
                % self.hparams.weights_init
            )

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

    def calculate_loss_generator(
        self, discr, total_cycle_consistency_loss, id, real, gen_id
    ):
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
        self.id_y = (
            self.generator_g2f(self.real_y) if self.hparams.lambda_idt != 0 else 0
        )
        self.id_x = (
            self.generator_f2g(self.real_x) if self.hparams.lambda_idt != 0 else 0
        )

        # Discriminate fake generated images
        self.discrimined_y = self.discriminator_y(self.fake_y)
        self.discrimined_x = self.discriminator_x(self.fake_x)

        # Discriminate real generated images
        self.discrimined_y_real = self.discriminator_y(self.real_y)
        self.discrimined_x_real = self.discriminator_x(self.real_x)

    def calculate_losses(self):
        total_cycle_consistency_loss = self.cycle_consistence_loss(
            self.rec_x, self.real_x
        ) + self.cycle_consistence_loss(self.rec_y, self.real_y)

        d_x_disc_fake = self.discriminator_x(self.fakePoolA.query(self.fake_x))
        d_y_disc_fake = self.discriminator_y(self.fakePoolB.query(self.fake_y))

        self.g_loss = self.calculate_loss_generator(
            self.discrimined_y,
            total_cycle_consistency_loss,
            self.id_y,
            self.real_y,
            "g",
        )
        self.f_loss = self.calculate_loss_generator(
            self.discrimined_x,
            total_cycle_consistency_loss,
            self.id_x,
            self.real_x,
            "f",
        )
        self.d_x_loss = self.calculate_loss_discriminator(
            self.discrimined_x_real, d_x_disc_fake, "x"
        )
        self.d_y_loss = self.calculate_loss_discriminator(
            self.discrimined_y_real, d_y_disc_fake, "y"
        )

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
        self.untoggle_optimizer(optimizer_g)

        # ---------------- TRAIN GENERATOR F ----------------
        self.toggle_optimizer(optimizer_f)
        optimizer_f.zero_grad()
        self.manual_backward(self.f_loss, retain_graph=True)
        self.untoggle_optimizer(optimizer_f)

        # ---------------- TRAIN OPTIMIZER DX ----------------
        self.toggle_optimizer(optimizer_d_x)
        optimizer_d_x.zero_grad()
        self.manual_backward(self.d_x_loss, retain_graph=False)
        self.untoggle_optimizer(optimizer_d_x)

        # ---------------- TRAIN OPTIMIZER DY ----------------
        self.toggle_optimizer(optimizer_d_y)
        optimizer_d_y.zero_grad()
        self.manual_backward(self.d_y_loss, retain_graph=False)
        self.untoggle_optimizer(optimizer_d_y)
        # ===================================================

        # update weights
        optimizer_f.step()
        optimizer_g.step()
        optimizer_d_x.step()
        optimizer_d_y.step()

        # update scheduler's step
        if (
            self.hparams.scheduler_enabled
            and batch_idx % self.hparams.scheduler_step_freq == 0
        ):
            scheduler_g.step()
            scheduler_f.step()
            scheduler_d_x.step()
            scheduler_d_y.step()
            print(f"scheduler: learning rate updated: G-{scheduler_g.get_last_lr()}")
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
            self.logger.experiment.add_image(
                "generated_images", grid, self.global_step / 4
            )

    def test_step(self, batch, batch_idx):
        self.te_real_x = batch[0][0].unsqueeze(0)
        self.te_fake_y = self.generator_g2f(self.te_real_x)
        self.te_rec_x = self.generator_f2g(self.te_fake_y)
        self.te_dis_real_x = self.discriminator_x(self.te_real_x)
        self.te_dis_fake_y = self.discriminator_y(self.te_fake_y)
        self.te_real_y = batch[0][1][0].unsqueeze(0).unsqueeze(0)

        self.te_fake_x = self.generator_f2g(self.te_real_y)
        self.te_rec_y = self.generator_g2f(self.te_fake_x)
        self.te_dis_real_y = self.discriminator_y(self.te_real_y)
        self.te_dis_fake_x = self.discriminator_x(self.te_fake_x)

        # calculate similarity for Image
        eval_metrics_y = self.image_evaluator(self.te_fake_y, self.te_real_y, "test-y")
        eval_metrics_x = self.image_evaluator(self.te_fake_x, self.te_real_x, "test-x")

        # log similarity metrics
        self.log_dict(eval_metrics_x)
        self.log_dict(eval_metrics_y)

        # log images
        grid = make_grid(
            [
                self.te_real_x[0],
                grayscale_to_rgb(self.te_fake_y[0]),
                self.te_rec_x[0],
                grayscale_to_rgb(self.te_real_y[0]),
                self.te_fake_x[0],
                grayscale_to_rgb(self.te_rec_y[0]),
            ],
            nrow=3,
        )
        image_folder = pathlib.Path("test-images-out")
        image_folder.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(grid, image_folder / f"{batch_idx}.png")

        # calculate the loss
        dx_loss_real = self.discriminator_loss(
            self.te_dis_real_x, torch.ones_like(self.te_dis_real_x)
        )
        dx_loss_fake = self.discriminator_loss(
            self.te_dis_fake_x, torch.zeros_like(self.te_dis_fake_x)
        )
        dx_loss = self.hparams.lambda_discriminator * (dx_loss_real + dx_loss_fake)
        dy_loss_real = self.discriminator_loss(
            self.te_dis_real_y, torch.ones_like(self.te_dis_real_y)
        )
        dy_loss_fake = self.discriminator_loss(
            self.te_dis_fake_y, torch.zeros_like(self.te_dis_fake_y)
        )
        dy_loss = self.hparams.lambda_discriminator * (dy_loss_real + dy_loss_fake)

        self.log_dict(
            {
                f"test_loss_Dx": dx_loss,
                f"test_loss_Dx_real": dx_loss_real,
                f"test_loss_Dx_fake": dx_loss_fake,
                f"test_loss_Dy": dy_loss,
                f"test_loss_Dy_real": dy_loss_real,
                f"test_loss_Dy_fake": dy_loss_fake,
            }
        )

    def validation_step(self, batch, batch_idx):
        self.v_real_x = batch[0][0].unsqueeze(0)
        self.v_fake_y = self.generator_g2f(self.v_real_x)
        self.v_rec_x = self.generator_f2g(self.v_fake_y)
        self.v_dis_real_x = self.discriminator_x(self.v_real_x)
        self.v_dis_fake_y = self.discriminator_y(self.v_fake_y)

        self.v_real_y = batch[0][1][0].unsqueeze(0).unsqueeze(0)
        self.v_fake_x = self.generator_f2g(self.v_real_y)
        self.v_rec_y = self.generator_g2f(self.v_fake_x)
        self.v_dis_real_y = self.discriminator_y(self.v_real_y)
        self.v_dis_fake_x = self.discriminator_x(self.v_fake_x)

        # calculate similarity for Image
        eval_metrics_y = self.image_evaluator(self.v_fake_y, self.v_real_y, "val-y")
        eval_metrics_x = self.image_evaluator(self.v_fake_x, self.v_real_x, "val-x")

        # log similarity metrics
        self.log_dict(eval_metrics_x)
        self.log_dict(eval_metrics_y)

        if batch_idx % self.hparams.log_nth_image == 0:
            # log images
            grid = make_grid(
                [
                    self.v_real_x[0],
                    grayscale_to_rgb(self.v_fake_y[0]),
                    self.v_rec_x[0],
                    grayscale_to_rgb(self.v_real_y[0]),
                    self.v_fake_x[0],
                    grayscale_to_rgb(self.v_rec_y[0]),
                ],
                nrow=3,
            )
            self.logger.experiment.add_image("val_generated_images", grid, batch_idx)

        # calculate the loss
        dx_loss_real = self.discriminator_loss(
            self.v_dis_real_x, torch.ones_like(self.v_dis_real_x)
        )
        dx_loss_fake = self.discriminator_loss(
            self.v_dis_fake_x, torch.zeros_like(self.v_dis_fake_x)
        )
        dx_loss = self.hparams.lambda_discriminator * (dx_loss_real + dx_loss_fake)
        dy_loss_real = self.discriminator_loss(
            self.v_dis_real_y, torch.ones_like(self.v_dis_real_y)
        )
        dy_loss_fake = self.discriminator_loss(
            self.v_dis_fake_y, torch.zeros_like(self.v_dis_fake_y)
        )
        dy_loss = self.hparams.lambda_discriminator * (dy_loss_real + dy_loss_fake)

        self.log_dict(
            {
                f"val_loss_Dx": dx_loss,
                f"val_loss_Dx_real": dx_loss_real,
                f"val_loss_Dx_fake": dx_loss_fake,
                f"val_loss_Dy": dy_loss,
                f"val_loss_Dy_real": dy_loss_real,
                f"val_loss_Dy_fake": dy_loss_fake,
            }
        )
