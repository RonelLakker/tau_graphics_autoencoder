import datetime
import torch
import random
import torch.nn as nn
import torchvision.datasets as dset
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# constants:

# Number of workers for dataloader
num_workers = 4

# Root directory for dataset
dataroot = "/home/ML_courses/03683533_2021/ronel_bar/dataset/"

# Validation size
validation_size = 1000

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 256

# Batch size during training
batch_size = 128

# Number of training epochs
num_epochs = 20

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # # # Encoder
            
            # starting with the 2d image tensor
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size is (16) * 128 * 128

            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size is (32) * 64 * 64

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size is (64) * 32 * 32

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size is (128) * 16 * 16

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size is (256) * 8 * 8

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size is (512) * 4 * 4

            nn.Conv2d(512, 256, 4, 1, 0),
            nn.ReLU(True),
            # state size is (256) * 1 * 1

            nn.Flatten(),
            # state size is [256]

            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
        )

        self.decoder = nn.Sequential(
            # # # Decoder

            nn.ConvTranspose2d(256, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size is (512) * 4 * 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size is (256) * 8 * 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size is (128) * 16 * 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size is (64) * 32 * 32

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size is (32) * 64 * 64

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size is (16) * 128 * 128
            
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        res = self.decoder(encoded.reshape([-1,256,1,1]))
        return res

    def decode(self, input):
        return self.decoder(input)


def init_data_loader():
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset

    # TODO: separate validation and load properly:

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


s = set()
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif "Conv" in classname or "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    s.add(classname)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("working on CPU!")
        device = torch.device("cpu")

    dataloader = init_data_loader()

    auto_enc = AutoEncoder().to(device)

    auto_enc.apply(weights_init)
    print(s)

    # Initialize BCELoss function (L2)
    criterion = nn.MSELoss()

    # Setup Adam optimizer
    optimizer = optim.Adam(auto_enc.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    losses = []
    iters = 0

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, 256, 1, 1, device=device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            auto_enc.zero_grad()
            # Format batch
            images = data[0].to(device)
            b_size = images.size(0)
            # Forward pass
            output = auto_enc(images).view(-1)
            # Calculate loss
            err = criterion(output, images)
            # Calculate gradients in backward pass
            err.backward()

            optimizer.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, num_epochs, i, len(dataloader), err.item()))

            # Save Loss for plotting later
            losses.append(err.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 5000 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = auto_enc.decode(fixed_noise).detach().cpu()
                
                vutils.save_image(fake, fp=f"test_images/image{iters//5000}.png", normalize=True, padding=2)
                print(f"saved image{iters//5000}.png")

            iters += 1

    torch.save(auto_enc.state_dict(), "model.pt")
    print(datetime.datetime.now())
