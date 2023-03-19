import numpy as np
from PIL import Image
import cv2

def tensor2im(image_tensor, imtype=np.uint8):

    image_numpy = image_tensor.cpu().float().numpy().squeeze(0)
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255

    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_tensor_img(img_tensor, img_path):
    img_numpy = tensor2im(img_tensor)
    save_image(img_numpy, img_path)
