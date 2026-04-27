import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Bag", "DGSI"]

class Bag(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low_gate = nn.Conv2d(channels, channels, 1)
        self.high_gate = nn.Conv2d(channels, channels, 1)

    def forward(self, p, i, d):
        # Use the current-scale feature to generate separate gates for low-level detail
        # and high-level semantic features, instead of a single complementary gate.
        low_att = torch.sigmoid(self.low_gate(d))
        high_att = torch.sigmoid(self.high_gate(d))
        gate_sum = low_att + high_att + 1e-6
        return (low_att / gate_sum) * p + (high_att / gate_sum) * i

class DGSI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.bag = Bag(out_features // 4)
        self.tail_conv = nn.Conv2d(out_features, out_features, 1)
        self.conv = nn.Conv2d(out_features // 2, out_features // 4, 1)
        self.bns = nn.BatchNorm2d(out_features)
        self.skips = nn.Conv2d(in_features[1], out_features, 1)
        self.skips_2 = nn.Conv2d(in_features[0], out_features, 1)
        self.skips_3 = nn.Conv2d(
            in_features[2], out_features, kernel_size=3, stride=2, dilation=2, padding=2
        )
        self.silu = nn.SiLU()

    def forward(self, x_list):
        x_low, x, x_high = x_list
        if x_high is not None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low is not None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(
                x_low, size=[x.size(2), x.size(3)], mode="bilinear", align_corners=True
            )
            x_low = torch.chunk(x_low, 4, dim=1)
        x = self.skips(x)
        x_skip = x
        x = torch.chunk(x, 4, dim=1)
        if x_high is None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low is None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.silu(x)
        return x
