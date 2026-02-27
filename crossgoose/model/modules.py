
from torch import nn

from crossgoose.cellpose import models as cp_models


class CPBackbone(nn.Module):
    def __init__(self, device, load_pretrained: bool = True):
        super().__init__()
        self.device = device

        self.nchan = 2
        nclasses = 3
        self.nbase = [32, 64, 128, 256]
        self.nbase = [self.nchan, *self.nbase]
        diam_mean = 30

        net = cp_models.CPnet(
            nbase=self.nbase, nout=nclasses, sz=3, mkldnn=False,
            max_pool=True, diam_mean=diam_mean).to(self.device)

        if load_pretrained:
            pretrained_model, diam_mean, _, _ = cp_models.get_model_params(
                pretrained_model='cyto3',
                model_type=None,
                pretrained_model_ortho=None,
                default_model='cyto3')

            net.load_model(pretrained_model, device=self.device)

        self.downsample = net.downsample
        self.make_style = net.make_style
        self.upsample = net.upsample
        # self.output = net.output

    def forward(self, data):

        c = data.shape[1]
        if c != self.nchan:
            raise ValueError(
                f"data.shape[1]={c} does not mach n_chan={self.nchan}")

        # the cellpose way
        T0 = self.downsample(data)

        style = self.make_style(T0[-1])

        T1 = self.upsample(style, T0, False)
        # T1 is of feature size 32
        # T2 = self.output(T1)

        return T1, style, T0


def linblock(in_channels, out_channels):
    return nn.Sequential(
        nn.ReLU(),
        nn.Linear(in_channels, out_channels),
    )
