import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import model
import matplotlib.pyplot as plt
import torchvision.utils as vutils


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

        fake = m.maybe_better_decode(fixed_noise).detach().cpu()

        vutils.save_image(fake, fp=output_path, normalize=True, padding=2)


if __name__ == '__main__':
    # forward_image_from_path("bar.jpeg","bar_output.png","model.pt")
    # forward_image_from_path("ronel.jpeg","ronel_output.png","model.pt")
    # forward_image_from_path("00000.png","00000out.png","model.pt")
    forward_random_latent_vectors("maybe_random_generate.2.25.png", "model.1635821666.2835324.25.pt")

