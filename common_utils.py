import cv2
import PIL
PIL.Image.init()
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import os
import functools
import torchvision as tv
import pathlib
import torch.nn.functional as F
import random
from typing import Any, List, Tuple, Union

resize = functools.partial(F.interpolate, mode = 'bicubic', align_corners = True)
downsample = functools.partial(F.interpolate, mode = 'bilinear', align_corners = True)
adaptive_pool = functools.partial(F.interpolate, mode = 'area')

def toggle_device(args, device):
    out = []
    for th in args:
        out.append(th.to(device))
    return out


def toggle_grad(models, flag = True):
    for model in models:
        try:
            for p in model.parameters():
                p.requires_grad = flag
        except:
            model.requires_grad = flag

def show_image(x, in_type = 'matplot', in_range = '0,1', **kwargs):
    x = tensor_array_pil(x, in_type = in_type, out_type = 'matplot', in_range = in_range, out_range = '0,1', **kwargs)
    plt.figure()
    plt.imshow(x)
    plt.show()


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def show_array_image(x, ):
    assert x.shape[-1] == 1 or x.shape[-1] == 3
    plt.figure()
    plt.imshow(x)
    plt.show()


def show_tensor_image(x, nrow = 1):
    x = ((x + 1) / 2).clamp(0, 1)
    x = torchvision.utils.make_grid(x, nrow = nrow, normalize = False)
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    plt.figure()
    plt.imshow(x)
    plt.show()


def check_shape(a, b):
    for i, s in enumerate(a.shape):
        if s != b[i]:
            raise Exception('Shape error, input shape {}, correct shape {}'.format(a.shape, b))


def lerp(w1, w2, beta):
    assert w1.shape == w2.shape
    w = w1 * beta + w2 * (1 - beta)
    return w


def make_dir(path, type = 'dir'):
    assert type in ['dir', 'path']
    path = pathlib.Path(path)
    if type == 'dir':
        path.mkdir(exist_ok = True, parents = True)

    if type == 'path':
        path.parent.mkdir(exist_ok = True, parents = True)


def torch_cat(args, dim = 0):
    if args[0] is None:
        return torch.cat(args[1:], dim = dim)
    else:
        return torch.cat(args, dim = dim)


def denorm(x):
    return (x + 1) / 2


def tensor_array_pil(image, in_type = 'tensor', out_type = 'pil', in_range = '0,1', out_range = '0,1'):
    '''
    image in [0,1]
    '''
    assert in_type in ['tensor', 'cv2', 'matplot', 'pil']
    assert out_type in ['tensor', 'cv2', 'matplot', 'pil']
    assert in_range in ['0,1', '-1,1', '0,255']
    assert out_range in ['0,1', '-1,1', '0,255']
    if in_type == 'tenor' and isinstance(image, torch.Tensor):
        if in_range == '0,255':
            image = torch.clip(image.to(torch.float) / 255, 0, 1)
        elif in_range == '-1,1':
            image = torch.clip((image + 1) / 2, 0, 1)
        if out_range == '0,255':
            image = (image * 255).to(torch.uint8)
        elif out_range == '-1,1':
            image = (image - 0.5) / 0.5

        if out_type == 'tensor':
            return image if image.ndim == 4 else image.unsqueeze(0)

        image = image.squeeze(0) if image.ndim == 4 else image
        image = image.detach().cpu().permute(1, 2, 0).numpy()

        if out_type == 'cv2':
            image = image[..., ::-1]
        elif out_type == 'pil':
            image = Image.fromarray(image)
        return image

    elif in_type == 'cv2' or 'matplot' or 'pil':
        image = np.array(image)
        if in_range == '0,1':
            image = image * 255
        elif in_range == '-1,1':
            image = 255 * ((image + 1) / 2)
        image = np.clip(image, 0, 255)
        tensor_image = image_to_tensor(image, in_type = in_type, out_ndim = 3, out_range = out_range)
        return tensor_array_pil(tensor_image, in_type = 'tensor', out_type = out_type, in_range = '0,255',
                                out_range = out_range)


def image_to_tensor(image, in_type = 'cv2', out_ndim = 4, in_range = '0,255', out_range = '-1,1'):
    """ input must range from 0 to 255"""
    assert in_type.lower() in ['cv2', 'pil', 'matplot']
    assert out_ndim in [3, 4]
    assert out_range in ['-1,1', '0,1', '0,255']

    if out_range == '-1,1':
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif out_range == '0,1':
        normalize = lambda x: x
    elif out_range == '0,255':
        normalize = lambda x: 255.0 * x

    tensor_img = eval(f'convert_{in_type}_to_tensor')(image, normalize = normalize)
    if out_ndim == 4:
        return torch.unsqueeze(tensor_img, 0)
    return tensor_img


def hwc_to_chw(hwc_image):
    return hwc_image.transpose(2, 0, 1)


def convert_PIL_to_tensor(pil_image, normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transform(pil_image)


def convert_cv2_to_tensor(bgr_image, normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
    rgb_image = bgr_image[..., ::-1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transform(rgb_image)


def convert_matplot_to_tensor(rgb_image, normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])):
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transform(rgb_image)


composed = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)])


def read_image(path, range = '-1,1', type = 'tensor', size = None):

    assert range in ['-1,1', '0,1', '0,255']
    assert type in ['array', 'tensor']

    img = np.ascontiguousarray(cv2.imread(path)[..., ::-1])

    if size is not None:
        img = cv2.resize(img, size)

    img = img.astype(np.float)

    if range == '0,1':
        img = img / 255
    elif range == '-1,1':
        img = (img - 127.5) / 127.5

    if type == 'array':
        return img
    elif type == 'tensor':
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def save_pth(file, path):
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok = True, parents = True)
    torch.save(file, path)


def save_image(img, path, nrow = 1, range = '-1,1', type = 'tensor', **kwargs):
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok = True, parents = True)
    assert range in ['0,1', '-1,1', '0,255']
    if range == '0,1':
        img = img
    if range == '-1,1':
        img = ((img + 1) / 2)
    if range == '0,255':
        img = img / 255

    # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()[0].numpy()[..., ::-1]
    img = img.clamp(0, 1)
    tv.utils.save_image(img, str(path), normalize = False, nrow = nrow)

# Load a model's weights
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


def show(img):
    plt.figure(figsize = (12, 8))
    npimg = img.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    plt.figure()
    plt.imshow(npimg)
    plt.show()


def seed_all(seed, deterministic = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_image_paths(image_paths):
    def recursive(image_paths, return_paths):
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                _all_fnames = [os.path.join(root, fname) for root, _dirs, files in os.walk(image_paths) for fname in
                               files]
                image_paths = sorted(
                    fname for fname in _all_fnames if os.path.splitext(fname)[1].lower() in PIL.Image.EXTENSION)
            elif os.path.exists(image_paths):
                image_paths = [image_paths]
            else:
                return
            return_paths += image_paths
        elif isinstance(image_paths, list):
            for image_path in image_paths:
                recursive(image_path, return_paths)

    return_paths = []
    # import pdb;
    # pdb.set_trace()
    recursive(image_paths, return_paths)
    return return_paths