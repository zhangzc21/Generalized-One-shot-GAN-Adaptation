import abc
import os
import pickle
from argparse import Namespace
import os.path
from models.e4e.psp import pSp
from torchvision import transforms
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import dnnlib
import dlib
import glob
from alignment import align_face
import lpips
from SSIM import MS_SSIM
from training.loss import CLIPLoss
import common_utils
def pre_process_images(raw_images_path, out_dir):
    IMAGE_SIZE = 1024
    predictor = dlib.shape_predictor(r'pretrained_models/align.dat')
    # os.chdir(raw_images_path)
    images_names = glob.glob(f'{raw_images_path}/*')

    aligned_images = []
    for image_name in tqdm(images_names):
        try:
            aligned_image = align_face(filepath = f'{image_name}',
                                       predictor = predictor, output_size = IMAGE_SIZE)
            aligned_images.append(aligned_image)
        except Exception as e:
            print(e)

    os.makedirs(out_dir, exist_ok = True)
    for image, name in zip(aligned_images, images_names):
        real_name = os.path.basename(name).split('.')[0]
        image.save(f'{out_dir}/{real_name}.jpeg')


def toogle_grad(model, flag = True):
    for p in model.parameters():
        p.requires_grad = flag


def load_old_G(path = 'pretrained_models/ffhq.pkl'):
    with open(path, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda().eval()
        old_G = old_G.float()
    return old_G


class ensemble_projector:
    def __init__(self, device = 'cuda', e4e_model = 'pretrained_models/e4e_ffhq_encode.pt'):
        self.use_last_w_pivots = False
        self.e4e_model = e4e_model
        self.w_pivots = {}
        self.image_counter = 0
        self.device = device

        self.first_inv_type = 'w+'

        if self.e4e_model is not None:
            self.initilize_e4e()
            self.e4e_image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def calc_inversions(self, image, mask = None):

        if self.e4e_model is not None:
            synth_image, w = self.get_e4e_inversion(image)
        elif mask is not None:
            synth_image, w = self.invert_face_with_mask(image, mask, num_steps = 1000, initial_w = None, out_dir = None,
                                  device = 'cuda')
        else:
            id_image = torch.squeeze((image.to(self.device) + 1) / 2) * 255
            synth_image, w = project(self.G, id_image, device = torch.device(self.device), w_avg_samples = 1000,
                        num_steps = 1000, w_name = None, mask = mask)

        return synth_image, w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self, learning_rate = 3e-4):
        optimizer = torch.optim.Adam(self.G.parameters(), lr = learning_rate)

        return optimizer

    def forward(self, w):
        generated_images = self.G.synthesis(w, noise_mode = 'const', force_fp32 = True)
        return generated_images

    def initilize_e4e(self):
        ckpt = torch.load(self.e4e_model, map_location = 'cpu')
        opts = ckpt['opts']
        opts['batch_size'] = 1
        opts['checkpoint_path'] = self.e4e_model
        opts = Namespace(**opts)
        self.e4e_inversion_net = pSp(opts)
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net = self.e4e_inversion_net.to(self.device)
        toogle_grad(self.e4e_inversion_net, False)

    def get_e4e_inversion(self, image):
        image = (image + 1) / 2
        new_image = self.e4e_image_transform(image[0].detach().cpu()).to(self.device)
        image, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise = False, return_latents = True,
                                          resize = False,
                                          input_code = False)
        return image, w

    def invert_face_with_mask(self, target, mask, num_steps = 1000, initial_w = None, out_dir = None, device = 'cuda'):
        w_avg_samples = 10000
        initial_learning_rate = 0.01
        initial_noise_factor = 0.05
        lr_rampdown_length = 0.25
        lr_rampup_length = 0.05
        noise_ramp_length = 0.7
        regularize_noise_weight = 1e5

        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(device).float()  # in_type: ignore

        if mask is not None:
            mask = F.interpolate(1- mask, size = (256, 256), mode = 'area')[:, :1, ...].to(torch.float32).to(device)
        else:
            mask = torch.ones(1, 1, 256, 256).to(torch.float32).to(device)

        # Compute w stats.
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis = 0, keepdims = True)  # [1, 1, C]
        w_avg_tensor = torch.from_numpy(w_avg).to(device)
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        start_w = initial_w if initial_w is not None else w_avg

        # Setup noise inputs.
        noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

        # Features for target image.
        # import pdb;pdb.set_trace()
        target_images = target.to(device).to(torch.float32)
        target_images_ = F.interpolate(target_images, size = (256, 256), mode = 'area')

        w_opt = torch.tensor(start_w, dtype = torch.float32, device = device,
                             requires_grad = True)  # pylint: disable=not-callable
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas = (0.9, 0.999),
                                     lr = 5e-3)
        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        clip_loss = CLIPLoss(device = device).to(torch.float32)
        ssim_loss = MS_SSIM(window_size = 11, window_sigma = 1.5, data_range = 1.0, channel = 3).to(torch.float32).to(device)
        lpips_loss = lpips.LPIPS(net = 'vgg').to(torch.float32).to(device).eval()
        pbar = tqdm(range(num_steps))
        for step in pbar:
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            synth_images = G.synthesis(ws, noise_mode = 'const', force_fp32 = True)

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images_ = F.interpolate(synth_images, size = (256, 256), mode = 'area')
            # import pdb;pdb.set_trace()
            ssim_dist = (1 - ssim_loss(common_utils.denorm(target_images_ * mask), common_utils.denorm(synth_images_ * mask))).mean()
            lpips_dist = lpips_loss(synth_images_ * mask, target_images_ * mask)
            clip_dist = clip_loss.rec_loss(synth_images_ * mask, target_images_ * mask)

            # Features for synth images.
            dist = ssim_dist + lpips_dist + clip_dist
            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts = 1, dims = 3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts = 1, dims = 2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size = 2)
            loss = dist + reg_loss * regularize_noise_weight

            loss_dict = dict(ssim_dist=ssim_dist, lpips_dist=lpips_dist, clip_dist=clip_dist, regularize_noise_weight= reg_loss)
            for key in loss_dict.keys():
                loss_dict[key] = float(loss_dict[key])

            pbar.set_postfix(**loss_dict)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()


            if (out_dir) is not  None and ((step + 1) % 200 == 0 or step == 0):
                if step == 0:
                    common_utils.save_image(target_images, path = os.path.join(out_dir, f'target.jpg'))
                common_utils.save_image(synth_images, path = os.path.join(out_dir, f'inv_{step}.jpg'))


            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        synth_image = G.synthesis(w_opt.repeat([1, G.num_ws, 1]), noise_mode = 'const', force_fp32 = True)
        w_opt = w_opt.repeat([1, G.num_ws, 1])
        del G
        return synth_image, w_opt

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""


def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic out_range [0,255], W & H must match G output resolution
        *,
        num_steps = 1000,
        w_avg_samples = 10000,
        initial_learning_rate = 0.01,
        initial_noise_factor = 0.05,
        lr_rampdown_length = 0.25,
        lr_rampup_length = 0.05,
        noise_ramp_length = 0.75,
        regularize_noise_weight = 1e5,
        verbose = False,
        device: torch.device,
        initial_w = None,
        mask = None,
        w_name: str
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # in_type: ignore

    if mask is not None:
        mask = F.interpolate(mask, size = (256, 256), mode = 'area')[:,:1,...].to(device)
    else:
        mask = torch.ones(1,1,256,256).to(device)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis = 0, keepdims = True)  # [1, 1, C]
    w_avg_tensor = torch.from_numpy(w_avg).to(device)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size = (256, 256), mode = 'area')
    target_features = vgg16(target_images* mask, resize_images = False, return_lpips = True)

    w_opt = torch.tensor(start_w, dtype = torch.float32, device = device,
                         requires_grad = True)  # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas = (0.9, 0.999),
                                 lr = 5e-3)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True


    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode = 'const', force_fp32 = True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size = (256, 256), mode = 'area')

        # Features for synth images.
        synth_features = vgg16(synth_images * mask, resize_images = False, return_lpips = True)
        dist = ((target_features - synth_features)).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts = 1, dims = 3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts = 1, dims = 2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size = 2)
        loss = dist + reg_loss * regularize_noise_weight

        # if step % image_log_step == 0:
        #     with torch.no_grad():
        #         if use_wandb:
        #             global_config.training_step += 1

        # Step
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    synth_image = G.synthesis(w_opt.repeat([1, G.num_ws, 1]), noise_mode = 'const', force_fp32 = True)
    w_opt = w_opt.repeat([1, G.num_ws, 1])
    del G
    return synth_image, w_opt
