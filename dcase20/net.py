import torch.nn as nn


def ActLayer(act):
    assert act in ['relu', 'leakyrelu', 'tanh'], 'Unknown activate function!'
    if act == 'relu':
        return nn.ReLU(True)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.2, True)
    elif act == 'tanh':
        return nn.Tanh()


def NormLayer(normalize, chan, reso):
    assert normalize in ['bn', 'ln', 'in'], 'Unknown normalize function!'
    if normalize == 'bn':
        return nn.BatchNorm2d(chan)
    elif normalize == 'ln':
        return nn.LayerNorm((chan, reso, reso))
    elif normalize == 'in':
        return nn.InstanceNorm2d(chan)


class DCEncoder(nn.Module):
    """
    DCGAN DCEncoder NETWORK
    """

    def __init__(self, isize, nz, ndf, act, normalize, add_final_conv=True):
        super(DCEncoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = []
        main.append(nn.Conv2d(1, ndf, 4, 2, 1, bias=False))
        main.append(NormLayer(normalize, ndf, isize // 2))
        main.append(ActLayer(act))
        csize, cndf = isize // 2, ndf

        while csize > 4:
            in_chan = cndf
            out_chan = cndf * 2
            main.append(nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False))
            cndf = cndf * 2
            csize = csize // 2
            main.append(NormLayer(normalize, out_chan, csize))
            main.append(ActLayer(act))

        # state size. K x 4 x 4
        if add_final_conv:
            main.append(nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = nn.Sequential(*main)

    def forward(self, x):
        z = self.main(x)
        return z


class DCDecoder(nn.Module):
    """
    DCGAN DCDecoder NETWORK
    """
    def __init__(self, isize, nz, ngf, act, normalize):
        super(DCDecoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = []
        main.append(nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        csize = 4
        main.append(NormLayer(normalize, cngf, csize))
        main.append(ActLayer(act))

        while csize < isize // 2:
            main.append(nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            cngf = cngf // 2
            csize = csize * 2
            main.append(NormLayer(normalize, cngf, csize))
            main.append(ActLayer(act))

        main.append(nn.ConvTranspose2d(cngf, 1, 4, 2, 1, bias=False))
        main.append(ActLayer('tanh'))
        self.main = nn.Sequential(*main)

    def forward(self, z):
        x = self.main(z)
        return x


class AEDC(nn.Module):
    def __init__(self, param):
        super(AEDC, self).__init__()
        self.Encoder = DCEncoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ndf=param['net']['ndf'],
                                 act=param['net']['act'][0],
                                 normalize=param['net']['normalize']['g'])
        self.Decoder = DCDecoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ngf=param['net']['ngf'],
                                 act=param['net']['act'][1],
                                 normalize=param['net']['normalize']['g'])

    def forward(self, data, outz=False):
        z = self.Encoder(data)
        if outz:
            return z
        else:
            recon = self.Decoder(z)
            return recon


class Discriminator(nn.Module):
    def __init__(self, param):
        super(Discriminator, self).__init__()
        ndf, isize = param['net']['ndf'], param['net']['isize']
        act, normalize = param['net']['act'][0], param['net']['normalize']['d']

        self.main = nn.ModuleList()
        level = 0
        in_chan = 1
        chans, resoes = [in_chan], [isize]
        init_layer = nn.Sequential(nn.Conv2d(in_chan, ndf, 4, 2, 1, bias=False),
                                   NormLayer(normalize, ndf, isize // 2),
                                   ActLayer(act))
        level, csize, cndf = 1, isize // 2, ndf
        self.main.append(init_layer)
        chans.append(ndf)
        resoes.append(csize)

        while csize > 4:
            in_chan = cndf
            out_chan = cndf * 2
            pyramid = [nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False)]
            level, cndf, csize = level + 1, cndf * 2, csize // 2
            pyramid.append(NormLayer(normalize, out_chan, csize))
            pyramid.append(ActLayer(act))
            self.main.append(nn.Sequential(*pyramid))
            chans.append(out_chan)
            resoes.append(csize)

        in_chan = cndf
        # 判断真假
        self.feat_extract_layer = nn.Sequential(nn.Conv2d(in_chan, in_chan, 4, 1, 0, bias=False, groups=in_chan),  # GDConv
                                                nn.Flatten())  # D网络的embedding
        self.output_layer = nn.Sequential(nn.LayerNorm(in_chan),
                                          ActLayer(act),
                                          nn.Linear(in_chan, 1))

    def forward(self, x):
        for module in self.main:
            x = module(x)
        feat = self.feat_extract_layer(x)
        pred = self.output_layer(feat)
        return pred, feat
