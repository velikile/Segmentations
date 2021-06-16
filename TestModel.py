import torch
from PIL import Image
from torchvision import transforms
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torch.load("mrcnn.m")
    img = transforms.ToTensor()(Image.open('imgX.png'))
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        print(prediction)
        imgmask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        imgmask.save('nnmask.png','PNG')
        transforms.ToPILImage()(img).convert('RGB').save('imgX.png','PNG')
if(__name__ == '__main__'):
    main()
