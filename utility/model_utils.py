from model.model import SiameseNetwork
import torch
import torchvision
from PIL import Image

new_model = SiameseNetwork()
new_model.load_state_dict(torch.load('model/siameseNetowrk_final.pt', map_location="cpu"))
new_model.eval()

def get_image_embedding(img_path):
    convert_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((92,112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
    embd=new_model.forward_128(convert_tensor(Image.open(img_path)).unsqueeze(0)).squeeze()
    return embd.detach().numpy()
