import json

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

def validate_first_batch(dataset_path, output_grid_path, model_path, hyper_params=model.models[0]):
    # Check how the generator is doing by saving G's output on fixed_noise
    model.hyper_params = hyper_params
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
def validate_on_dataset(file_pathes, model_path, hyper_params=model.models[0]):
    if len(file_pathes) > 128:
        print("big file path, truncating to 128 samples!")
        file_pathes = file_pathes[:128]
    model.hyper_params = hyper_params
    m = model.AutoEncoder()
    m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    m.eval()
    for image_path in file_pathes:
        tensor_img = load_image_as_tensor(image_path)
        output_image = m(tensor_img.unsqueeze(0))
        vutils.save_image(torch.cat((tensor_img.unsqueeze(0),output_image)), fp=image_path+"__comparison.png", normalize=True, padding=2)


def show_losses(losses_paths, labels=None, title="Loss During Training"):
    if labels is None:
        labels = [f"model{i}" for i in range(len(losses_paths))]
    models_losses = [json.loads(open(losses_path, "r").read()) for losses_path in losses_paths]
    # models_losses = [[losses[i] for i in range(len(losses)) if i % (1500//(len(losses)**0.001)) == 0] for losses in models_losses]
    plt.figure(figsize=(10, 5))
    plt.title(title)
    for label, losses in zip(labels, models_losses):
        plt.plot([i/len(losses) for i in range(len(losses))], losses, label=label)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.gca().set_ylim([0, 0.2])
    plt.legend()
    plt.show()


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


def forward_image_from_path(image_path, output_image_path, model_path, hyper_params=model.models[0]):
    with torch.no_grad():
        model.hyper_params = hyper_params
        m = model.AutoEncoder()
        m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        m.eval()
        tensor_img = load_image_as_tensor(image_path)
        output_image = m(tensor_img.unsqueeze(0))

        vutils.save_image(torch.cat((tensor_img.unsqueeze(0),output_image)), fp=output_image_path, normalize=True, padding=2)


def bar_forward_random_latent_vectors(output_path, model_path, hyper_params=model.models[0]):
    with torch.no_grad():
        model.hyper_params = hyper_params
        m = model.AutoEncoder()
        m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        m.eval()

        encoded = torch.load("bar_encoded.pt", map_location=torch.device("cpu"))

        fixed_noise = torch.normal(0, 10, size=[64, 256, 1, 1], device='cpu')
        fixed_noise += encoded

        fake = m.decode(fixed_noise).detach().cpu()

        vutils.save_image(fake, fp=output_path, normalize=True, padding=2)

def forward_random_latent_vectors(output_path, model_path, hyper_params=model.models[0]):
    with torch.no_grad():
        model.hyper_params = hyper_params
        m = model.AutoEncoder()
        m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        m.eval()

        fixed_noise = torch.normal(0, 10, [64, 256, 1, 1], device='cpu')

        fake = m.decode(fixed_noise).detach().cpu()

        vutils.save_image(fake, fp=output_path, normalize=True, padding=2)

def interpolate_tensors(t1_path, t2_path, output_path, model_path, hyper_params=model.models[0]):
    with torch.no_grad():
        model.hyper_params = hyper_params
        m = model.AutoEncoder()
        m.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        m.eval()

        encoded1 = torch.load(t1_path, map_location=torch.device("cpu"))
        encoded2 = torch.load(t2_path, map_location=torch.device("cpu"))

        fixed_noise = torch.normal(0, 5, size=[64, 256, 1, 1], device='cpu')
        fixed_noise += (encoded1 + encoded2) / 2

        fake = m.decode(fixed_noise).detach().cpu()

        vutils.save_image(fake, fp=output_path, normalize=True, padding=2)


if __name__ == '__main__':
    # forward_image_from_path("bar.jpeg", "bar3_output.png", "./model3.2021-11-03_01_59_38.pt", hyper_params=model.models[3])
    # forward_image_from_path("ronel.jpeg","ronel3_output.png","./model3.2021-11-03_01_59_38.pt", hyper_params=model.models[3])
    # for i in range(7):
    #     forward_image_from_path("bar.jpeg", f"bar{i}_output.png", f"./model{i}.2021-11-03_01_59_38.pt", hyper_params=model.models[i])
    #     forward_image_from_path("ronel.jpeg", f"ronel{i}_output.png", f"./model{i}.2021-11-03_01_59_38.pt", hyper_params=model.models[i])
    # forward_image_from_path("00000.png","00000out.png","model.pt")
    # forward_random_latent_vectors("rand_gen_.png", "./model0.2021-11-02_22_44_43.pt")
    # for i in range(7):
    #     forward_random_latent_vectors(f"rand_gen_model{i}.png", f"./model{i}.2021-11-03_01_59_38.pt", hyper_params=model.models[i])
    # interpolate_tensors("ronel_encoded.pt", "bar_encoded.pt", "rand_gen_22_44_43.png", "./model0.2021-11-02_22_44_43.pt")
    # for j in range(7):
    #     show_losses([f"model{i}_losses2021-11-03_01_59_38.data" for i in [j]], labels=[f"model{i}" for i in [j]])
    # show_losses([f"model{i}_losses2021-11-03_01_59_38.data" for i in range(7)], labels=[f"model{i}" for i in range(7)])
    # show_losses([f"model{i}_validation_losses2021-11-03_01_59_38.data" for i in range(7)], labels=[f"model{i}" for i in range(7)], title="Validation Loss")

    losses_paths = [f"model{i}_validation_losses2021-11-03_01_59_38.data" for i in range(7)]
    models_losses = [json.loads(open(losses_path, "r").read()) for losses_path in losses_paths]
    models_avgs = [(i,sum(models_losses[i])/len(models_losses[i])) for i in range(7)]
    models_avgs.sort(key=lambda x: x[1])
    for i, avg in models_avgs:
        print(f"model{i}'s validation loss avg: {avg}")

