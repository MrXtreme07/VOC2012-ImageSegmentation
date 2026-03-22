import torch
from PIL import Image
import torchvision.transforms as transforms
from models.unet import UNet
import matplotlib.pyplot as plt
from utils.visualize import decode_segmap

device = torch.device('cpu')

model = UNet(num_classes=21)
checkpoint = torch.load('checkpoints/model_epoch_198.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return image, pred

if __name__ == '__main__':
    image, mask = predict('dataset/VOC2012/JPEGImages/2007_000027.jpg')

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')

    color_mask = decode_segmap(mask)
    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.title('Segmentation')

    plt.show()
