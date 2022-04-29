import torch
import torchvision
from PIL import Image

#######
from dataloadercache.cache import CacheImage
#######
pascalvoc=torchvision.datasets.VOCDetection(root="dataset/",download=True)

imgcache = CacheImage()

original_img_open=Image.open

def imgopen(imgcache,path):
    if imgcache.shared_dict is None:
        return original_img_open(path)

    if path in imgcache.shared_dict:
        return original_img_open(imgcache.shared_dict[path])
    else:

        with open(path, 'rb') as fh:
            buf = BytesIO(fh.read())

        if len(imgcache.shared_dict)<imgcache.max_elements:
            imgcache.shared_dict[path] = buf
    
    return original_img_open(buf)
Image.open=imgopen