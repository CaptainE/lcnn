import torch
from torchvision import transforms
from dataset import NYUDataset
from custom_transforms import *
import plot_utils
import model_utils
from torchvision.utils import save_image

#Info

#3x640x480 in dataset,   CxWxH
#480x640x3 for plotting, HxWxC
#3x480x640 for pytorch,  CxHxW


def main():
    bs = 8
    sz = (512,512)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean, std = torch.tensor(mean), torch.tensor(std)
    unnormalize = UnNormalizeImgBatch(mean, std)

    tfms = transforms.Compose([
        ResizeImgAndDepth(sz),
        ImgAndDepthToTensor()#,      
        #NormalizeImg(mean, std)
    ])

    ds = NYUDataset('/home/pebert/lcnn/', tfms)
    dl = torch.utils.data.DataLoader(ds, bs, shuffle=True)
    for i in range(len(ds)):
        pth = 'utils/images/img_depth' + str(i)+ '.jpg'
        if i>200:
            save_image(ds[i][1], pth)
        if i >220:
            break

# to save depth change name: _depth and do ds[i][1]/255



    #i = 1
    #plot_utils.plot_image(model_utils.get_unnormalized_ds_item(unnormalize, ds[i]))

if __name__ == "__main__":
    main()
