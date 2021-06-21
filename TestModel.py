import torch
from PIL import Image
from torchvision import transforms
import numpy as np

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torch.load("mrcnn.m")
    img = transforms.ToTensor()(Image.open('CoffeeBeanValidation/0i.png'))
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        imgmask   = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        maskcolor = np.stack((imgmask,)*3,axis=-1)
        rand_colors = np.random.randint(0,255,size=(1000,3))
        for m in range(prediction[0]['masks'].shape[0]):
            pred = prediction[0]['masks'][m, 0].mul(255).byte().cpu().numpy()
            pred = np.stack((pred,)*3,axis=-1)
            r_color = rand_colors[m%100]
            r = pred[:,:,0]
            g = pred[:,:,1]
            b = pred[:,:,2]
            r[r!=0] = r_color[0]
            g[g!=0] = r_color[1]
            b[b!=0] = r_color[2]

            maskcolor = maskcolor| pred
            imgmask = Image.fromarray(maskcolor)

        imgmask.convert('RGB').save('nnmask.png','PNG')
        transforms.ToPILImage()(img).convert('RGB').save('imgX.png','PNG')

if(__name__ == '__main__'):
    main()
