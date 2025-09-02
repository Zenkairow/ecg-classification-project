# data_synthesis/src/gan_model.py

import torch
import torch.nn as nn

class UNetDown(nn.Module):
    """A U-Net encoder block."""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """A U-Net decoder block with skip connections."""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    """The U-Net Generator for our Pix2Pix GAN."""
    def __init__(self, in_channels=3, out_channels=1):
        super(Generator, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        # Add more downsampling layers for higher resolution images if needed
        
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(512, 128) # Concatenated size
        self.up3 = UNetUp(256, 64)  # Concatenated size
        
        # Final layer to produce the output
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final_up(u3)

class Discriminator(nn.Module):
    """The PatchGAN Discriminator."""
    def __init__(self, in_channels=1): # Takes a single-channel signal image as input
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)

# --- Block for testing the model architectures ---
if __name__ == '__main__':
    # Create a dummy input image tensor
    # batch_size=4, channels=3 (RGB), height=256, width=512
    dummy_image = torch.randn(4, 3, 256, 512)
    
    # Create instances of the models
    generator = Generator()
    discriminator = Discriminator()
    
    print("--- Generator Architecture Test ---")
    generated_signal_image = generator(dummy_image)
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Generated signal image shape: {generated_signal_image.shape}") # Should have 1 channel
    
    print("\n--- Discriminator Architecture Test ---")
    discriminator_output = discriminator(generated_signal_image)
    print(f"Discriminator input shape: {generated_signal_image.shape}")
    print(f"Discriminator output (patch) shape: {discriminator_output.shape}")