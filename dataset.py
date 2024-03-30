from os import listdir
from os.path import join
import random
from PIL import Image ,ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import is_image_file
import torch
import cv2


def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray
def transf(im):
  im = im.resize((256, 256), Image.BICUBIC)
  im=transforms.ToTensor()(im)
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
