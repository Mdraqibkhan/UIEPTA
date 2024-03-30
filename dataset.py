from os import listdir
from os.path import join
import random
from PIL import Image ,ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import is_image_file
# from skimage import io, color
import torch
import cv2
from pytorch_msssim import ssim
# from torchvision.transforms.functional import pad
# rgb = io.imread(filename)
# lab = color.rgb2lab(rgb)

def torchPSNR(tar_img, prd_img):
  imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
  rmse = (imdff**2).mean().sqrt()
  ps = 20*torch.log10(1/rmse)
  return ps

def torchSSIM(tar_img, prd_img):
  return ssim(tar_img, prd_img, data_range=1.0, size_average=True)

# def save_img(filepath, img):
#     cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# def pad_img(img, padded_size):
#     C, H, W = img.shape
#     pad_size = [(padded_size[0]-H) // 2, (padded_size[1]-W) // 2]
#     return pad(img, pad_size)    


def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

# def rgb_2_lab(im):
#    # srgb_p = ImageCms.createProfile("sRGB")
#    # lab_p  = ImageCms.createProfile("LAB")
#    # rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
#    # Lab = ImageCms.applyTransform(im, rgb2lab)
#    return color.rgb2lab(im)

def transf(im):
  im = im.resize((256, 256), Image.BICUBIC)
  im=transforms.ToTensor()(im)
  # w_offset = random.randint(0, max(0, 286 - 256 - 1))
  # h_offset = random.randint(0, max(0, 286 - 256 - 1))
  # im = im[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
  im = transforms.Normalize((0.5),(0.5))(im)
  return im 

  
class DatasetFromFolder(data.Dataset):
  def __init__(self, image_dir):
    super(DatasetFromFolder, self).__init__()
    self.a_path = join(image_dir, "a")
    self.b_path = join(image_dir, "b")
    self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]


  def __getitem__(self, index):

    a1 = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
    tar1 = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
  
    in_grey= a1.convert('L')
    in_grey = in_grey.resize((256, 256), Image.BICUBIC)
    in_grey=transforms.ToTensor()(in_grey)
    # in_grey = transforms.Normalize((0.5,0.5),(0.5,0.5))(in_grey)

    
    transform_list = [transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)

    a1 = a1.resize((256,256), Image.BICUBIC)
    tar1 = tar1.resize((256,256), Image.BICUBIC)
    a1 =transform(a1)
    tar1 = transform(tar1)
    
    return a1,tar1,in_grey

  def __len__(self):
    return len(self.image_filenames)