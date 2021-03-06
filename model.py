import json
import datetime
import time
import torch
import random
import torch.nn as nn
import torchvision.datasets as dset
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils

manualSeed = 1000
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# constants:

models = [
    # M1:
    {'loss': nn.MSELoss,
     'batch_norm': True,
     'biases': False,
     'stride': 2,
     'batch_size': 128,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 0},

    # M2:
    {'loss': nn.L1Loss,
     'batch_norm': True,
     'biases': False,
     'stride': 2,
     'batch_size': 128,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.LeakyReLU(0.02),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 0},

    # M3:
    {'loss': nn.MSELoss,
     'batch_norm': False,
     'biases': True,
     'stride': 4,
     'batch_size': 128,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 0},

    # M4:
    {'loss': nn.MSELoss,
     'batch_norm': False,
     'biases': False,
     'stride': 2,
     'batch_size': 16,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 0},

    # M5:
    {'loss': nn.MSELoss,
     'batch_norm': True,
     'biases': False,
     'stride': 2,
     'batch_size': 64,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 0},

    # M6:
    {'loss': nn.MSELoss,
     'batch_norm': True,
     'biases': False,
     'stride': 2,
     'batch_size': 64,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.LeakyReLU(0.02),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 0},

    # M7:
    {'loss': nn.MSELoss,
     'batch_norm': True,
     'biases': False,
     'stride': 2,
     'batch_size': 128,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 1},

    # M8:
    {'loss': nn.MSELoss,
     'batch_norm': True,
     'biases': False,
     'stride': 2,
     'batch_size': 128,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.SGD(parameters, lr=lr),
     'fc_layers': 0},

    # M9
    {'loss': nn.MSELoss,
     'batch_norm': False,
     'biases': False,
     'stride': 2,
     'batch_size': 128,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=lr, betas=(beta1, 0.999)),
     'fc_layers': 0},

    # M10:
    {'loss': nn.MSELoss,
     'batch_norm': True,
     'biases': False,
     'stride': 2,
     'batch_size': 128,
     'activation_Enc': lambda: nn.LeakyReLU(0.02),
     'activation_Dec': lambda: nn.ReLU(True),
     'optimizer': lambda parameters: optim.Adam(parameters, lr=0.0001, betas=(beta1, 0.999)),
     'fc_layers': 0},
]

iter_models = iter(models)

hyper_params = next(iter_models)

# Number of workers for dataloader
num_workers = 4

# Root directory for dataset
dataroot = "/home/ML_courses/03683533_2021/ronel_bar/dataset/"

# Validation size
validation_size = 1024

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 256

# # Batch size during training
# batch_size = 128

# Number of training epochs

num_epochs = 50 #2

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

ngpu = 5


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.hyper_params = hyper_params

        self.enc_active = self.hyper_params['activation_Enc']
        self.dec_active = self.hyper_params['activation_Dec']

        self.encoder = nn.Sequential(
            # # # Encoder
            
            # starting with the 2d image tensor
            nn.Conv2d(3, 16, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(16) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.enc_active(),
            # state size is (16) * 128 * 128

            nn.Conv2d(16, 32, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(32) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.enc_active(),
            # state size is (32) * 64 * 64

            nn.Conv2d(32, 64, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(64) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.enc_active(),
            # state size is (64) * 32 * 32

            nn.Conv2d(64, 128, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(128) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.enc_active(),
            # state size is (128) * 16 * 16

            nn.Conv2d(128, 256, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(256) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.enc_active(),
            # state size is (256) * 8 * 8

            nn.Conv2d(256, 512, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(512) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.enc_active(),
            # state size is (512) * 4 * 4

            nn.Conv2d(512, 256, 4, 1, 0, bias=self.hyper_params['biases']),
            self.enc_active(),
            # state size is (256) * 1 * 1

            nn.Flatten(),
            # state size is [256]
        )

        self.fully_connected = nn.Sequential(
            # # # Decoder
            *tuple([item for sublist in [[nn.Linear(256, 256), nn.ReLU(True)] for _ in range(self.hyper_params['fc_layers'])] for item in sublist])
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 4, 1, 0, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(512) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.dec_active(),
            # state size is (512) * 4 * 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(256) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.dec_active(),
            # state size is (256) * 8 * 8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(128) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.dec_active(),
            # state size is (128) * 16 * 16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(64) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.dec_active(),
            # state size is (64) * 32 * 32

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(32) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.dec_active(),
            # state size is (32) * 64 * 64

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.BatchNorm2d(16) if self.hyper_params['batch_norm'] else nn.Identity(),
            self.dec_active(),
            # state size is (16) * 128 * 128
            
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=self.hyper_params['biases']),
            nn.Tanh(),
        )

    def forward(self, input):
        encoded = self.fully_connected(self.encoder(input))
        # torch.save(encoded.reshape([-1,256,1,1])[0], "ronel_encoded.pt")
        res = self.decoder(encoded.reshape([-1,256,1,1]))
        return res

    def decode(self, input):
        return self.decoder(input)
        # return self.decoder(self.fully_connected(input.reshape([-1,256])).reshape([-1,256,1,1]))


def init_data_loader(batch_size):
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
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


s = set()
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif "Conv" in classname or "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    s.add(classname)


if __name__ == '__main__':
    start_time = str(datetime.datetime.now()).replace(" ", "_")[:19]
    f = open(f"log{start_time}", "a")
    def log(p):
        f.write(str(p) + "\n")
        f.flush()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        log("working on CPU!")
        device = torch.device("cpu")
    
    model_index = 0
    
    while hyper_params is not None:
        try:
            auto_enc_data = AutoEncoder().to(device)

            dataloader = init_data_loader(hyper_params['batch_size'])

            auto_enc = nn.DataParallel(auto_enc_data, list(range(ngpu)))

            auto_enc.apply(weights_init)
            log(s)

            # Initialize loss function
            criterion = hyper_params['loss']()

            # Setup Adam optimizer
            optimizer = hyper_params['optimizer'](auto_enc.parameters())

            # Lists to keep track of progress
            losses = []
            iters = 0
            start_date = datetime.datetime.now()

            # Create batch of latent vectors that we will use to visualize
            #  the progression of the generator
            fixed_noise = torch.randn(64, 256, 1, 1, device=device)
            validation_images = next(iter(dataloader))[0]
            validation_images_gpu = next(iter(dataloader))[0].to(device)

            log(f"Starting Training Loop... {start_date}")
            # For each epoch
            for epoch in range(num_epochs):
                # For each batch in the dataloader
                for i, data in enumerate(dataloader, 0):
                    if i < validation_size / hyper_params['batch_size']:
                        continue
                    auto_enc.zero_grad()
                    # Format batch
                    images = data[0].to(device)
                    b_size = images.size(0)
                    # Forward pass
                    output = auto_enc(images)
                    # Calculate loss
                    err = criterion(output, images)
                    # Calculate gradients in backward pass
                    err.backward()

                    optimizer.step()

                    # Output training stats
                    if i % 50 == 0:
                        log('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, num_epochs, i, len(dataloader), err.item()))

                    # Save Loss for plotting later
                    losses.append(err.item())

                    # # Check how the generator is doing by saving G's output on fixed_noise
                    # freq = num_epochs * 12 * 2
                    # if (iters % freq == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    #     auto_enc.eval()
                    #     for j in range(validation_images.size(0)//32):
                    #         test_samples = validation_images_gpu[j*32:j*32+32]
                    #         with torch.no_grad():
                    #             output_samples = auto_enc(test_samples).detach().cpu()
                    #         results = torch.cat((validation_images[j*32:j*32+32], output_samples))
                    #
                    #         vutils.save_image(results, fp=f"test_images/image{start_time}.{iters//freq}.{j}.png", normalize=True, padding=2)
                    #         log(f"saved image{start_time}.{iters//freq}.{j}.png")
                    #     auto_enc.train()
                    #     torch.save(auto_enc.state_dict(), f"model.{start_time}.{iters//freq}.pt")

                    # iters += 1

            ### evaluation
            validation_losses = []

            auto_enc.eval()
            test_samples = validation_images_gpu[0:32]
            with torch.no_grad():
                output_samples = auto_enc(test_samples).detach().cpu()
            results = torch.cat((validation_images[0:32], output_samples))

            vutils.save_image(results, fp=f"test_images/image_model{model_index}.png", normalize=True, padding=2)
            log(f"saved model test image")

            with torch.no_grad():
                for i, data in enumerate(dataloader, 0):
                    if i >= validation_size / hyper_params['batch_size']:
                        break
                    # Format batch
                    images = data[0].to(device)
                    b_size = images.size(0)
                    # Forward pass
                    output = auto_enc(images)
                    # Calculate loss
                    err = criterion(output, images)

                    # Save Loss for plotting later
                    validation_losses.append(err.item())

            auto_enc.train()

            torch.save(auto_enc_data.state_dict(), f"model{model_index}.{start_time}.pt")
            log(f"saved model as: 'model{model_index}.{start_time}.pt'")
            open(f"model{model_index}_losses{start_time}.data", "w").write(json.dumps(losses))
            open(f"model{model_index}_validation_losses{start_time}.data", "w").write(json.dumps(validation_losses))
        except Exception as e:
            log(f"Exception in model{model_index}!")
            log(str(e))
        end_date = datetime.datetime.now()
        log(f"finished model{model_index} in {end_date - start_date}")
        
        hyper_params = next(iter_models)
        model_index += 1

    f.close()
