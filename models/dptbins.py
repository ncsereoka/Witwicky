import torch
import torch.nn as nn

from .dpt import DPTDepthModel
from .miniViT import mViT


class DptBins(nn.Module):
    def __init__(self, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(DptBins, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.dpt_base = DPTDepthModel(
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

    def forward(self, x, **kwargs):
        dpt_out = self.dpt_base(x, **kwargs)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(dpt_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.dpt_base.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        print('Building DPT-based model..', end='')
        m = cls(n_bins=n_bins, **kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    model = DptBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
