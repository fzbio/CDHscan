"""The CDHscan network: a Hi-C map encoder and a WGS track encoder joined before a small head.

ResidualEncoder turns one stack of 100 x 100 Hi-C channels into 128 numbers (2D residual CNN),
or one stack of 100-bin WGS tracks, with the same layout in 1D. JointCNN uses one
or both encoders; their numbers are joined end to end and a small head maps them to one logit.
"""

import torch
from torch import nn

STAGE_CHANNELS = [16, 32, 64, 128]   # one stage per entry; stages 2-4 halve the length: 100 -> 50 -> 25 -> 13
BLOCKS_PER_STAGE = 2
HEAD_HIDDEN = 64
DROPOUT = 0.3


class ResidualBlock(nn.Module):
    """Two size-3 convolutions with batch normalisation, plus a shortcut that adds the input back.
    dimensions = 2 for contact maps, 1 for tracks."""

    def __init__(self, in_channels, out_channels, stride, dimensions):
        super().__init__()
        conv = nn.Conv2d if dimensions == 2 else nn.Conv1d
        norm = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm1d
        self.body = nn.Sequential(
            conv(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm(out_channels), nn.ReLU(),
            conv(out_channels, out_channels, 3, padding=1, bias=False),
            norm(out_channels))
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(conv(in_channels, out_channels, 1, stride=stride, bias=False),
                                          norm(out_channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))


class ResidualEncoder(nn.Module):
    """Stem convolution, then 4 stages of residual blocks (17 convolution layers), then global average pooling."""

    def __init__(self, n_input_channels, dimensions):
        super().__init__()
        conv = nn.Conv2d if dimensions == 2 else nn.Conv1d
        norm = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm1d
        pool = nn.AdaptiveAvgPool2d(1) if dimensions == 2 else nn.AdaptiveAvgPool1d(1)
        layers = [conv(n_input_channels, STAGE_CHANNELS[0], 3, padding=1, bias=False), norm(STAGE_CHANNELS[0]), nn.ReLU()]
        in_channels = STAGE_CHANNELS[0]
        for stage, out_channels in enumerate(STAGE_CHANNELS):
            for block in range(BLOCKS_PER_STAGE):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResidualBlock(in_channels, out_channels, stride, dimensions))
                in_channels = out_channels
        layers += [pool, nn.Flatten()]
        self.layers = nn.Sequential(*layers)
        self.output_size = STAGE_CHANNELS[-1]

    def forward(self, x):
        """x: (batch, n_input_channels, 100, 100) or (batch, n_input_channels, 100) -> (batch, 128)."""
        return self.layers(x)


class JointCNN(nn.Module):
    """n_map_channels = 0: no map encoder; n_track_channels = 0: no track encoder."""

    def __init__(self, n_map_channels, n_track_channels):
        super().__init__()
        self.map_encoder = ResidualEncoder(n_map_channels, dimensions=2) if n_map_channels else None
        self.track_encoder = ResidualEncoder(n_track_channels, dimensions=1) if n_track_channels else None
        n_features = sum(encoder.output_size for encoder in (self.map_encoder, self.track_encoder) if encoder is not None)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(n_features, HEAD_HIDDEN), nn.ReLU(),
                                  nn.Dropout(DROPOUT), nn.Linear(HEAD_HIDDEN, 1))

    def forward(self, contact_map, tracks):
        """Returns one logit per window; the input of an encoder the model does not have is None."""
        features = []
        if self.map_encoder is not None:
            features.append(self.map_encoder(contact_map))
        if self.track_encoder is not None:
            features.append(self.track_encoder(tracks))
        return self.head(torch.cat(features, dim=1)).squeeze(1)
