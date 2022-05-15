# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import torch.nn.functional as F
import functools
import random


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain,
                             cur_nimg):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe = None, r1_gamma = 10, style_mixing_prob = 0, pl_weight = 0,
                 pl_batch_shrink = 2, pl_decay = 0.01, pl_no_weight_grad = False, blur_init_sigma = 0,
                 blur_fade_kimg = 0):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device = device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

    def run_G(self, z, c, update_emas = False):
        ws = self.G.mapping(z, c, update_emas = update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype = torch.int64, device = ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device = ws.device) < self.style_mixing_prob, cutoff,
                                     torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas = False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas = update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma = 0, update_emas = False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device = img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas = update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3),
                         0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma = blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/lpips_loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(
                        self.pl_no_weight_grad):
                    pl_grads = \
                        torch.autograd.grad(outputs = [(gen_img * pl_noise).sum()], inputs = [gen_ws],
                                            create_graph = True,
                                            only_inputs = True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas = True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma = blur_sigma, update_emas = True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma = blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/lpips_loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs = [real_logits.sum()], inputs = [real_img_tmp],
                                                create_graph = True,
                                                only_inputs = True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()


# ----------------------------------------------------------------------------

def get_feature2(G, ws):
    block_ws = []
    self = G.synthesis
    feats = []
    with torch.autograd.profiler.record_function('split_ws'):
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

    x = img = None
    for res, cur_ws in zip(self.block_resolutions, block_ws):
        block = getattr(self, f'b{res}')
        x, img = block(x, img, cur_ws, noise_mode = 'const', force_fp32 = True)
        feats.append(x)
    return feats


def manifold_loss(self, z, G_old, G_new):
    num_samples = z.shape[0]
    z_samples1 = torch.randn_like(z)
    z_samples2 = torch.randn_like(z)

    directions1 = F.normalize(z_samples1 - z, dim = -1)
    directions2 = F.normalize(z_samples2 - z, dim = -1)
    # z1 = beta * z.expand(num_samples, *z.shape[1:]) + (1 - beta) * z_samples

    beta1 = random.random()
    beta2 = random.random()

    z1_lerp1 = z
    z1_lerp2 = z + beta1 * directions1
    z1_lerp3 = z + beta2 * directions2

    with torch.no_grad():
        feat1_source = get_feature2(G_old, z1_lerp1)[:9]
        feat2_source = get_feature2(G_old, z1_lerp2)[:9]
        feat3_source = get_feature2(G_old, z1_lerp3)[:9]

    feat1_target = get_feature2(G_new, z1_lerp1)[:9]
    feat2_target = get_feature2(G_new, z1_lerp2)[:9]
    feat3_target = get_feature2(G_new, z1_lerp3)[:9]

    diff1_source = list(map(lambda x, y: x - y, feat2_source, feat1_source))
    diff2_source = list(map(lambda x, y: x - y, feat3_source, feat1_source))

    diff1_target = list(map(lambda x, y: x - y, feat2_target, feat1_target))
    diff2_target = list(map(lambda x, y: x - y, feat3_target, feat1_target))

    sim_source = list(map(lambda x, y: torch.cosine_similarity(x, y, eps = 1e-6), diff1_source, diff2_source))
    sim_target = list(map(lambda x, y: torch.cosine_similarity(x, y, eps = 1e-6), diff1_target, diff2_target))

    loss_list = list(map(lambda x, y: torch.mean(F.smooth_l1_loss(x, y)), sim_source, sim_target))
    loss = functools.reduce(lambda x, y: x + y, loss_list) / 9

    return loss


import clip
from torchvision import transforms
from collections import deque


class CLIPLoss(torch.nn.Module):
    def __init__(self, device = 'cuda'):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device = device)
        resize = functools.partial(F.interpolate, size = (224, 224), mode = 'bicubic', align_corners = True)
        denormalize = lambda x: (x + 1) / 2
        self.tensor_preprocess = transforms.Compose([resize, denormalize,
                                                     transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                          (0.26862954, 0.26130258, 0.27577711))])
        self.d_ref = None
        self.d_queue = deque([], maxlen = 100)

        self.source_fixed_feature = None
        self.target_fixed_feature = None

    def simple_VLapR(self, source_sample, target_sample, L, normalize = False):
        source_feature = self.encoder(source_sample)
        target_feature = self.encoder(target_sample)
        if normalize:
            target_feature = F.normalize(target_feature, p = 2, dim = -1)
            source_feature = F.normalize(source_feature, p = 2, dim = -1)
        N = len(source_feature)
        L = L.to(torch.float16)
        diff = target_feature - source_feature
        M = ((diff.permute(1, 0)) @ L) @ diff
        loss = torch.trace(M)
        return loss / ((N ** 2 - N) * target_feature.shape[-1])

    def feature_forward(self, image_feature1, image_feature2):
        similarity = 1 - F.cosine_similarity(image_feature1, image_feature2, eps = 1e-6)
        return torch.mean(similarity)

    def VLapR(self, source_fixed, source_sample, target_fixed, target_sample, L, normalize = False, detach = True):
        if detach:
            with torch.no_grad():
                source = torch.cat([source_fixed, source_sample], dim = 0)
                source_feature = self.encoder(source).detach() # detach is very important
                target_fixed_feature = self.encoder(target_fixed).detach()
        else:
            source = torch.cat([source_fixed, source_sample], dim = 0)
            source_feature = self.encoder(source)
            target_fixed_feature = self.encoder(target_fixed)
        target_sample_feature = self.encoder(target_sample)
        target_feature = torch.cat([target_fixed_feature, target_sample_feature], dim = 0)
        if normalize:
            target_feature = F.normalize(target_feature, p = 2, dim = -1)
            source_feature = F.normalize(source_feature, p = 2, dim = -1)
        N = len(source)
        L = L.to(torch.float16)
        diff = target_feature - source_feature
        M = ((diff.permute(1,0)) @ L) @ diff
        loss = torch.trace(M)
        return loss / ((N**2-N)*target_feature.shape[-1])

    def forward(self, image1, image2):
        d_similarity = 1 - F.cosine_similarity(self.encoder(image1), self.encoder(image2), eps = 1e-6)
        return d_similarity

    def encoder(self, image):
        image = self.tensor_preprocess(image)
        return self.model.encode_image(image)

    def rec_loss(self, image1, image2):
        d_similarity = 1 - F.cosine_similarity(self.encoder(image1), self.encoder(image2), eps = 1e-6)
        return d_similarity

    def laplacian(self, ws, normalize = True):
        if normalize:
            eye = torch.eye(len(ws)).to(ws.device)
            W = torch.exp(-(ws.unsqueeze(1) - ws.unsqueeze(0)).norm(dim = -1, p = 2) / 128) - eye
            W = torch.from_numpy(np.around(W.cpu().numpy(), decimals = 4)).to(W.device)
            sqrt_D = torch.diag(torch.sum(W, dim = -1).pow(-0.5))
            L = eye - sqrt_D @ W @ sqrt_D
        else:
            W = torch.exp(-(ws.unsqueeze(1) - ws.unsqueeze(0)).norm(dim = -1, p = 2) / 128)
            D = torch.diag(torch.sum(W, dim = -1))
            L = D - W
        return L
