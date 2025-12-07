import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Блок остаточных связей с оптимизацией"""

    def __init__(self, in_channels, use_dropout=False, dropout_prob=0.5):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=True),
        )

        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Генератор с улучшенной архитектурой"""

    def __init__(
        self, in_channels=3, out_channels=3, num_residual_blocks=9, use_dropout=False
    ):
        super().__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        for i in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(
                    in_features, out_features, 3, stride=2, padding=1, bias=False
                ),
                nn.InstanceNorm2d(out_features, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features, use_dropout)]

        # Upsampling
        for i in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_features, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """Дискриминатор PatchGAN"""

    def __init__(self, in_channels=3, base_channels=64, n_layers=3):
        super().__init__()

        model = []

        # Первый слой
        model += [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Промежуточные слои
        channels = base_channels
        for n in range(1, n_layers):
            channels_prev = channels
            channels = min(base_channels * (2**n), 512)
            model += [
                nn.Conv2d(channels_prev, channels, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Предпоследний слой
        model += [
            nn.Conv2d(channels, channels, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Выходной слой
        model += [nn.Conv2d(channels, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class CycleGANModel(nn.Module):
    """Обертка для всех моделей CycleGAN"""

    def __init__(self, config):
        super().__init__()
        self.G_AB = Generator(
            config.in_channels, config.out_channels, config.num_residual_blocks
        ).to(config.device)
        self.G_BA = Generator(
            config.in_channels, config.out_channels, config.num_residual_blocks
        ).to(config.device)
        self.D_A = Discriminator(config.in_channels).to(config.device)
        self.D_B = Discriminator(config.in_channels).to(config.device)

        print(
            f"[Model] Generator params: {sum(p.numel() for p in self.G_AB.parameters()) / 1e6:.2f}M"
        )
        print(
            f"[Model] Discriminator params: {sum(p.numel() for p in self.D_A.parameters()) / 1e6:.2f}M"
        )

    def forward(self, real_A, real_B):
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        return fake_A, fake_B
