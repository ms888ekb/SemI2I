import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, st, first=False):
        super(ConvBlock, self).__init__()
        self.first = first
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=st, padding=1,
                              padding_mode='reflect')
        # self.ln = nn.BatchNorm2d(out_ch)
        self.ln = nn.InstanceNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        y = self.ln(x)
        y = self.relu(y)
        return y, x


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, st, last=False):
        super(UpConvBlock, self).__init__()
        if not last:
            self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, st, st, padding_mode='reflect'),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2))
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, ks, st, st, padding_mode='reflect'),
                nn.Tanh()
            )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, st):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=st, padding=(1, 1),
                               padding_mode='reflect'),
                      nn.InstanceNorm2d(out_ch),
                      nn.ReLU(True),
                      nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=ks, stride=st, padding=(1, 1),
                                     padding_mode='reflect'),
                      nn.InstanceNorm2d(out_ch)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class EncoderSemI2I(nn.Module):
    def __init__(self, in_ch):
        super(EncoderSemI2I, self).__init__()
        self.conv1 = ConvBlock(4, 32, 3, 1)
        self.conv2 = ConvBlock(32, 64, 3, 2)
        self.conv3 = ConvBlock(64, 128, 3, 2)
        self.resblock1 = ResidualBlock(128, 128, 3, 1)
        self.resblock2 = ResidualBlock(128, 128, 3, 1)
        self.resblock3 = ResidualBlock(128, 128, 3, 1)

    def forward(self, x):
        x, skip = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x[0])

        x = nn.ReLU()(self.resblock1(x[0]))
        x = nn.ReLU()(self.resblock2(x))
        x = nn.ReLU()(self.resblock3(x))

        return x, skip


class DecoderSemI2I(nn.Module):
    def __init__(self, out_ch):
        super(DecoderSemI2I, self).__init__()
        self.unconv1 = UpConvBlock(128+32, 64, 3, 1)
        self.unconv2 = UpConvBlock(64+32, 32, 3, 1)
        self.unconv3 = UpConvBlock(32+32, out_ch, 3, 1, True)

    def forward(self, x, skip):
        inp_x1 = nn.UpsamplingNearest2d(scale_factor=.25)(skip)
        x = torch.cat((x, inp_x1), dim=1)
        x = self.unconv1(x)
        inp_x2 = nn.UpsamplingNearest2d(scale_factor=.5)(skip)
        x = torch.cat((x, inp_x2), dim=1)
        x = self.unconv2(x)
        inp_x3 = nn.UpsamplingNearest2d(scale_factor=1)(skip)
        x = torch.cat((x, inp_x3), dim=1)
        x = self.unconv3(x)
        return x


class AdaINVal(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3))) / (
                        x.shape[2] * x.shape[3]))

    def forward(self, x, style):
        return (style[1]*((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + style[0]).permute([2, 3, 0, 1])


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3))) / (
                        x.shape[2] * x.shape[3]))

    def forward(self, x, y):
        return (self.sigma(y) * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x)) + self.mu(y)).permute([2, 3, 0, 1])


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.5)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class CalulateMuSigma(nn.Module):
    def __init__(self):
        super(CalulateMuSigma, self).__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt(
            (torch.sum((x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2, (2, 3))) / (
                        x.shape[2] * x.shape[3]))

    def forward(self, x):
        m = self.mu(x)
        s = self.sigma(x)
        return m, s


if __name__ == "__main__":
    pass
