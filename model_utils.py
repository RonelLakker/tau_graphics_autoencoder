import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import model
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.datasets as dset

def init_data_loader(dataroot):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset

    # TODO: separate validation and load properly:

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(256),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=8, pin_memory=True)

def validate_first_batch(dataset_path, output_grid_path, model_path):
    # Check how the generator is doing by saving G's output on fixed_noise
    m = model.AutoEncoder()
    m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    m.eval()
    data_loader = init_data_loader(dataset_path)
    validation_images = next(iter(data_loader))
    for j in range(validation_images.size(0)//32):
        test_samples = validation_images[j*32:j*32+32]
        with torch.no_grad():
            output_samples = m(test_samples)
        results = torch.cat((validation_images[j*32:j*32+32], output_samples))
        
        vutils.save_image(results, fp=output_grid_path, normalize=True, padding=2)

# forward pass on images in file_pathes, save the comparison result in <original-path-with-ending>_comparison.png
def validate_on_dataset(file_pathes, model_path):
    if len(file_pathes) > 128:
        print("big file path, truncating to 128 samples!")
        file_pathes = file_pathes[:128]
    m = model.AutoEncoder()
    m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    m.eval()
    for image_path in file_pathes:
        tensor_img = load_image_as_tensor(image_path)
        output_image = m(tensor_img.unsqueeze(0))
        vutils.save_image(torch.cat((tensor_img.unsqueeze(0),output_image)), fp=image_path+"__comparison.png", normalize=True, padding=2)
    


def load_image_as_tensor(image_path):
    image = Image.open(image_path)
    transormation = transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(256),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])
    tensor_img = transormation(image)
    return tensor_img


def forward_image_from_path(image_path, output_image_path, model_path):
    with torch.no_grad():
        m = model.AutoEncoder()
        m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        m.eval()
        tensor_img = load_image_as_tensor(image_path)

        output_image = m(tensor_img.unsqueeze(0))

        vutils.save_image(torch.cat((tensor_img.unsqueeze(0),output_image)), fp=output_image_path, normalize=True, padding=2)


def forward_random_latent_vectors(output_path, model_path):
    with torch.no_grad():
        m = model.AutoEncoder()
        m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        m.eval()

        fixed_noise = torch.randn(64, 256, 1, 1, device='cpu')

        fake = m.decode(fixed_noise).detach().cpu()

        vutils.save_image(fake, fp=output_path, normalize=True, padding=2)


if __name__ == '__main__':
    # forward_image_from_path("bar.jpeg","bar_output.png","model1635860295.2771986.pt")
    forward_image_from_path("ronel.jpeg","ronel_output.png","model1635860295.2771986.pt")
    # forward_image_from_path("00000.png","00000out.png","model.pt")
    #forward_random_latent_vectors("rand_gen_27.png", "model1635860295.2771986.pt")

