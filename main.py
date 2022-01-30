from net.networks import Discriminator, GANLoss, CalulateMuSigma, EncoderSemI2I, AdaIN, DecoderSemI2I, AdaINVal
import torch
import torch.nn as nn
from torch.utils import data
from data.dataloader import GeoDataLoaderStd, GeoDataLoaderWithName
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import itertools
import kornia
import os
from osgeo import gdal
import argparse


class SemI2I:
    def __init__(self, args):
        # Define the model's modules:
        self.net_A_enc = EncoderSemI2I(in_ch=4)
        self.net_A_dec = DecoderSemI2I(out_ch=4)
        self.net_B_enc = EncoderSemI2I(in_ch=4)
        self.net_B_dec = DecoderSemI2I(out_ch=4)
        self.calculator = CalulateMuSigma()
        if args.model_mode == 'train':
            self.adain = AdaIN()
        else:
            self.adainval = AdaINVal()

        self.net_d_A = Discriminator(input_nc=4, norm_layer=nn.InstanceNorm2d)
        self.net_d_B = Discriminator(input_nc=4, norm_layer=nn.InstanceNorm2d)

        self.m_glob_A = 0
        self.s_glob_A = 0
        self.m_glob_B = 0
        self.s_glob_B = 0
        self.d_rate = 0.95

        # Define the losses:
        self.cross_rec_criterion = nn.L1Loss()
        self.self_rec_criterion = nn.L1Loss()
        self.gradient_criterion = nn.L1Loss()
        self.adv_criterion = GANLoss(use_lsgan=True).cuda()

        self.learning_rate_g = args.learning_rate_G
        self.learning_rate_d = args.learning_rate_D
        self.total_steps = args.max_iter
        self.decay_step = int(self.total_steps * 0.5)
        # Define optimizers:
        params = list(self.net_A_enc.parameters()) + list(self.net_A_dec.parameters()) + \
                 list(self.net_B_enc.parameters()) + list(self.net_B_dec.parameters())

        self.optimizer_g = torch.optim.Adam(params, lr=self.learning_rate_g, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(itertools.chain(self.net_d_A.parameters(), self.net_d_B.parameters()),
                                            lr=self.learning_rate_d, betas=(0.5, 0.999))

        if args.model_mode == 'train':
            self.source_data_loader = data.DataLoader(GeoDataLoaderStd(x_data=args.source,
                                                                       size=(int(args.input_size_source[0]), int(args.input_size_source[1]))),
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=True)

            self.target_data_loader = data.DataLoader(GeoDataLoaderStd(x_data=args.target,
                                                                       size=(int(args.input_size_target[0]), int(args.input_size_target[1]))),
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=True)

        else:
            self.val_data_loader = data.DataLoader(GeoDataLoaderWithName(x_data=args.source,
                                                                         size=(int(args.input_size_source[0]), int(args.input_size_source[1]))),
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   drop_last=True)

        self.writer = SummaryWriter(log_dir=os.path.join('runs/SemI2I/', args.name))
        self.ename = args.name
        os.makedirs(os.path.join('SemI2I/', self.ename), exist_ok=True)

    def lr_decay(self, base_lr, iter, max_iter, decay_step):
        return base_lr * ((max_iter - float(iter)) / float((max_iter - decay_step)))

    def adjust_learning_rate_G(self, optimizer, i_iter):
        lr = self.lr_decay(self.learning_rate_g, i_iter, self.total_steps, self.decay_step)
        optimizer.param_groups[0]['lr'] = lr

    def adjust_learning_rate_D(self, optimizer, i_iter):
        lr = self.lr_decay(self.learning_rate_d, i_iter, self.total_steps, self.decay_step)
        optimizer.param_groups[0]['lr'] = lr

    def set_inputs(self, x, y):
        self.real_A = x.cuda()
        self.real_B = y.cuda()

    def forward(self):
        # Get fakeA and fakeB:
        self.feat_A, skip_A = self.net_A_enc(self.real_A)
        self.feat_B, skip_B = self.net_B_enc(self.real_B)

        self.fake_B = self.net_B_dec(self.adain(self.feat_A, self.feat_B), skip_A)
        self.fake_A = self.net_A_dec(self.adain(self.feat_B, self.feat_A), skip_B)

        # Get self-reconstructed A and B:
        self.self_rec_A = self.net_A_dec(self.feat_A, skip_A)
        self.self_rec_B = self.net_B_dec(self.feat_B, skip_B)

        # Get cross-reconstructed A and B:
        fake_B_feat, fake_B_skip = self.net_B_enc(self.fake_B)
        fake_A_feat, fake_A_skip = self.net_A_enc(self.fake_A)

        self.cross_rec_A = self.net_A_dec(self.adain(fake_B_feat, fake_A_feat), fake_B_skip)
        self.cross_rec_B = self.net_B_dec(self.adain(fake_A_feat, fake_B_feat), fake_A_skip)

    def backward_G(self):
        # Train generator:

        # Self-reconstruction loss:
        self.self_rec_loss = self.self_rec_criterion(self.real_A, self.self_rec_A) + \
                             self.self_rec_criterion(self.real_B, self.self_rec_B)
        # Cross-reconstruction loss:
        self.cross_rec_loss = self.cross_rec_criterion(self.real_A, self.cross_rec_A) + \
                              self.cross_rec_criterion(self.real_B, self.cross_rec_B)

        # Gradinet edge loss:
        edge_real_A = kornia.filters.spatial_gradient(self.real_A, mode='sobel', order=1, normalized=True)
        edge_fake_B = kornia.filters.spatial_gradient(self.fake_B, mode='sobel', order=1, normalized=True)
        edge_real_B = kornia.filters.spatial_gradient(self.real_B, mode='sobel', order=1, normalized=True)
        edge_fake_A = kornia.filters.spatial_gradient(self.fake_A, mode='sobel', order=1, normalized=True)

        self.edge_loss = self.gradient_criterion(edge_real_A, edge_fake_B) + \
            self.gradient_criterion(edge_real_B, edge_fake_A)

        # Adversarial loss for G_A and G_B:
        self.gen_adv_loss_A = self.adv_criterion(self.net_d_A(self.fake_B), True)
        self.gen_adv_loss_B = self.adv_criterion(self.net_d_B(self.fake_A), True)

        self.gen_loss = 1 * (self.gen_adv_loss_A + self.gen_adv_loss_B) + 20 * self.cross_rec_loss +\
                        10 * self.self_rec_loss + 25 * self.edge_loss
        self.gen_loss.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.adv_criterion(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.adv_criterion(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B.detach()
        self.loss_D_A = self.backward_D_basic(self.net_d_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A.detach()
        self.loss_D_B = self.backward_D_basic(self.net_d_B, self.real_A, fake_A)

    def optimize_parameters(self):
        # Forward pass:
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        self.set_requires_grad([self.net_d_A, self.net_d_B], False)
        self.forward()
        self.backward_G()
        self.optimizer_g.step()

        self.set_requires_grad([self.net_d_A, self.net_d_B], True)
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_d.step()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_model(self):
        self.calculator.train(False)
        self.net_A_enc.train()
        self.net_B_enc.train()
        self.net_A_dec.train()
        self.net_B_dec.train()
        self.net_d_A.train()
        self.net_d_B.train()
        self.net_A_enc.cuda()
        self.net_B_enc.cuda()
        self.net_A_dec.cuda()
        self.net_B_dec.cuda()
        self.net_d_A.cuda()
        self.net_d_B.cuda()

        target_iterator = enumerate(self.target_data_loader)
        source_iterator = enumerate(self.source_data_loader)

        for step in range(self.total_steps):
            if step >= self.decay_step:
                self.adjust_learning_rate_G(self.optimizer_g, step)
                self.adjust_learning_rate_D(self.optimizer_d, step)

            # Get source batch:
            try:
                _, source_batch = next(source_iterator)
            except StopIteration:
                source_iterator = enumerate(self.source_data_loader)
                _, source_batch = next(source_iterator)

            # Get target batch:
            try:
                _, target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = enumerate(self.target_data_loader)
                _, target_batch = next(target_iterator)

            # Unpack batches:
            source_images, s_geo_ref = source_batch
            target_images, t_geo_ref = target_batch

            self.set_inputs(source_images, target_images)
            self.optimize_parameters()

            if step % 100 == 0:
                real_a = self.back_transformation(self.real_A)
                fake_b = self.back_transformation(self.fake_B)

                real_b = self.back_transformation(self.real_B)
                fake_a = self.back_transformation(self.fake_A)

                self_rec_a = self.back_transformation(self.self_rec_A)
                self_rec_b = self.back_transformation(self.self_rec_B)

                cross_rec_a = self.back_transformation(self.cross_rec_A)
                cross_rec_b = self.back_transformation(self.cross_rec_B)

                self.writer.add_image('real_A', real_a, step)
                self.writer.add_image('fake_B', fake_b, step)
                self.writer.add_image('real_B', real_b, step)
                self.writer.add_image('fake_A', fake_a, step)
                self.writer.add_image('self_rec_A', self_rec_a, step)
                self.writer.add_image('self_rec_B', self_rec_b, step)
                self.writer.add_image('cross_rec_A', cross_rec_a, step)
                self.writer.add_image('cross_rec_B', cross_rec_b, step)
                # self.informer()

            if step % 5 == 0:
                # self.writer.add_image('self_rec_A', srecA)
                # self.informer()
                self.writer.add_scalar('cross_rec_loss', self.cross_rec_loss, step)
                self.writer.add_scalar('self_rec_loss', self.self_rec_loss, step)
                self.writer.add_scalar('gen_adv_loss_A', self.gen_adv_loss_A, step)
                self.writer.add_scalar('gen_adv_loss_B', self.gen_adv_loss_B, step)
                self.writer.add_scalar('disc_adv_loss_A', self.loss_D_A, step)
                self.writer.add_scalar('disc_adv_loss_B', self.loss_D_B, step)
                self.writer.add_scalar('edge_loss', self.edge_loss, step)

            feat_A, feat_B = self.feat_A.detach(), self.feat_B.detach()
            self.m_glob_A = self.d_rate * self.m_glob_A + (1 - self.d_rate) * self.calculator(feat_A)[0]
            self.s_glob_A = self.d_rate * self.s_glob_A + (1 - self.d_rate) * self.calculator(feat_A)[1]
            self.m_glob_B = self.d_rate * self.m_glob_B + (1 - self.d_rate) * self.calculator(feat_B)[0]
            self.s_glob_B = self.d_rate * self.s_glob_B + (1 - self.d_rate) * self.calculator(feat_B)[1]

            if step == self.total_steps-1:
                print(f'Saving the model at step {step+1}...')
                torch.save(self.net_A_enc.state_dict(),
                           os.path.join(*['SemI2I/', self.ename + '/SemI2I_enc_A.pth']))
                torch.save(self.net_B_enc.state_dict(),
                           os.path.join(*['SemI2I/', self.ename + '/SemI2I_enc_B.pth']))
                torch.save(self.net_A_dec.state_dict(),
                           os.path.join(*['SemI2I/', self.ename + '/SemI2I_dec_A.pth']))
                torch.save(self.net_B_dec.state_dict(),
                           os.path.join(*['SemI2I/', self.ename + '/SemI2I_dec_B.pth']))
                torch.save(self.net_d_A.state_dict(),
                           os.path.join(*['SemI2I/', self.ename + '/SemI2I_disc_A.pth']))
                torch.save(self.net_d_B.state_dict(),
                           os.path.join(*['SemI2I/', self.ename + '/SemI2I_disc_B.pth']))

                # Save global styles:
                torch.save(self.m_glob_A,
                           os.path.join(*['SemI2I/', self.ename + '/global_style_A_m.pth']))
                torch.save(self.s_glob_A,
                           os.path.join(*['SemI2I/', self.ename + '/global_style_A_s.pth']))
                torch.save(self.m_glob_B,
                           os.path.join(*['SemI2I/', self.ename + '/global_style_B_m.pth']))
                torch.save(self.s_glob_B,
                           os.path.join(*['SemI2I/', self.ename + '/global_style_B_s.pth']))

    def test_and_save(self, output_folder=None):
        net_A_enc_restored = EncoderSemI2I(in_ch=4).cuda()
        net_B_dec_restored = DecoderSemI2I(out_ch=4).cuda()

        saved_encoder_A = os.path.join(*['SemI2I/', self.ename + '/SemI2I_enc_A.pth'])
        saved_decoder_B = os.path.join(*['SemI2I/', self.ename + '/SemI2I_dec_B.pth'])

        style_B_m = os.path.join(*['SemI2I/', self.ename + '/global_style_B_m.pth'])
        style_B_s = os.path.join(*['SemI2I/', self.ename + '/global_style_B_s.pth'])

        net_A_enc_restored.load_state_dict(torch.load(saved_encoder_A))
        net_B_dec_restored.load_state_dict(torch.load(saved_decoder_B))
        style_B_m = torch.load(style_B_m, map_location=lambda storage, loc: storage.cuda(0))
        style_B_s = torch.load(style_B_s, map_location=lambda storage, loc: storage.cuda(0))

        style_B = [style_B_m, style_B_s]
        net_A_enc_restored.eval()
        net_B_dec_restored.eval()

        output_folder = os.path.join(*['SemI2I/', self.ename + '/fake_B'])
        os.makedirs(output_folder, exist_ok=True)

        for i, val_data in enumerate(self.val_data_loader):
            if i % 25 == 0:
                print(f'{i} images processed ')
            images, georef, crs, name = val_data
            images = images.cuda()
            fake_A_feat, skip_A = net_A_enc_restored(images)
            fake_B = net_B_dec_restored(self.adainval(fake_A_feat, style_B), skip_A)
            fake_B = self.back_transformation(fake_B, all=True)

            # Save predicted labels:
            output = np.asarray(fake_B, dtype=np.uint8)

            dst = os.path.join(output_folder, name[0])

            drv = gdal.GetDriverByName('GTiff')
            dst_ds = drv.Create(dst, output.shape[1], output.shape[0], output.shape[2], gdal.GDT_Byte,
                                   options=['COMPRESS=DEFLATE'])
            dst_ds.SetGeoTransform(georef)
            dst_ds.SetProjection(crs[0])

            dst_ds.GetRasterBand(1).WriteArray(output[:, :, 0])
            dst_ds.GetRasterBand(2).WriteArray(output[:, :, 1])
            dst_ds.GetRasterBand(3).WriteArray(output[:, :, 2])
            dst_ds.GetRasterBand(4).WriteArray(output[:, :, 3])
            dst_ds = None

        print(f'Translated source images were saved to {output_folder}')

    def back_transformation(self, image_tensor, all=False):
        img = image_tensor.detach()
        image_numpy = img[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        if all:
            return image_numpy.astype(np.uint8)
        else:
            image_numpy = image_numpy[:, :, :3]
            image_numpy = image_numpy[:, :, ::-1]
            image_numpy = image_numpy.transpose(2, 0, 1)
            return image_numpy.astype(np.uint8)


def get_arguments(params):
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SemI2I Network")
    parser.add_argument("--source", type=str, default=params['x_source'],
                        help="Define a path to the source images.")
    parser.add_argument("--target", type=str, default=params['x_target'],
                        help="Define a path to the source images.")
    parser.add_argument("--batch-size", type=int, default=params['batch_size'],
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--max_iter", type=int, default=params['iters'],
                        help="Number of training steps.")
    parser.add_argument("--input-size-source", type=str, default=params['s_size'],
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--input-size-target", type=str, default=params['t_size'],
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate-G", type=float, default=0.0001,
                        help="Base learning rate for generator.")
    parser.add_argument("--learning-rate-D", type=float, default=0.00002,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--model-mode", type=str, default=params['mode'],
                        help="Choose the model mode: train/test")
    parser.add_argument("--name", type=str, default=params['experiment_name'],
                        help="Define the name of the experiment.")
    return parser.parse_args()


if __name__ == "__main__":
    params = {
        's_size': (512, 512),
        't_size': (512, 512),
        'batch_size': 1,
        'x_source': r"/data/BC5_full",
        'x_target': r"/data/AB3_full",
        'mode': 'train',
        'experiment_name': 'default'
    }

    args = get_arguments(params)

    model = SemI2I(args)

    if args.model_mode == 'train':
        model.train_model()
    else:
        model.test_and_save()
