import torch
import torch.nn as nn

import my_blocks
import common_utils
import torch.nn.functional as F
import functools
import numpy as np
from torch.utils.data._utils.collate import default_collate

def slowly_forward(f, batch, *args, **input_dict):
    l = len(batch)
    if torch.is_tensor(batch):
        is_tensor = True
    elif isinstance(batch, list):
        is_tensor = False
    else:
        print("input must be tensor or list")
        raise

    resutls = []
    for i in range(l):
        if is_tensor:
            input = batch[[i]]
        else:
            input = batch[i]
        result = f(input, *args, **input_dict)
        result = [r.squeeze(0) if torch.is_tensor(r) else r for r in result]
        resutls.append(result)
    results = default_collate(resutls)
    if len(results) == 1:
        return results[0]
    else:
        return results


class MaskStyleGAN(nn.Module):
    def __init__(self, G, ref, index = 8):
        super(MaskStyleGAN, self).__init__()
        self.synthesis = G.synthesis
        self.mapping = G.mapping
        self.aux = my_blocks.UNet__(512, 512 + 1, input_res = 32, output_res = 256, n=3, fix_channel = 64)  # Auxiliaries(n_classes = 4)
        self.register_buffer('style', ref)
        self.index = index
        self.num_ws = G.num_ws

    def mix_style(self, w, alpha = 0):
        ref = self.style.repeat(len(w), 1, 1)
        mask = alpha * torch.ones(len(w), self.num_ws, 512).to(w.device)
        mask[:, :self.index, :] = 1.0
        w = w * mask + ref * (1 - mask)
        return w

    def synthesis_forward(self, w, return_feature = None,  mix_style = True, alpha = 0.0, **kwargs):
        if mix_style is True:
            w = self.mix_style(w, alpha = alpha)
        block_ws = []
        feat = []
        images = []
        with torch.autograd.profiler.record_function('split_ws'):
            with torch.autograd.profiler.record_function('split_ws'):
                ws = w.to(torch.float32)
                w_idx = 0
                for res in self.synthesis.block_resolutions:
                    block = getattr(self.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.synthesis.block_resolutions, block_ws):
            block = getattr(self.synthesis, f'b{res}')
            x, img = block(x, img, cur_ws, **kwargs)
            if return_feature is not None and int(res) in return_feature:
                feat.append(x)

        returns = [img]
        if return_feature is not None:
            returns.append(feat)

        if len(returns) > 1:
            return returns

        return img

    def synthesis_aux_forward(self, w, mask_gt = None, return_feature = None, alpha = 0.0,
                              mix_style = True,
                              **kwargs):
        if mix_style is True:
            w = self.mix_style(w, alpha = alpha)
        block_ws = []
        feat = []
        feat_ = []
        with torch.autograd.profiler.record_function('split_ws'):
            with torch.autograd.profiler.record_function('split_ws'):
                ws = w.to(torch.float32)
                w_idx = 0
                for res in self.synthesis.block_resolutions:
                    block = getattr(self.synthesis, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

        x = img = None

        i = 0
        for res, cur_ws in zip(self.synthesis.block_resolutions, block_ws):
            block = getattr(self.synthesis, f'b{res}')
            x, img = block(x, img, cur_ws, **kwargs)
            if int(res) == self.aux.input_res:
                t_feature, t_mask = self.aux(x)
                if mask_gt is not None:
                    t_mask_ = common_utils.resize(mask_gt, t_feature.shape[-2:])
                else:
                    t_mask_ = common_utils.resize(t_mask, t_feature.shape[-2:])
                x = x + (t_feature - x) * t_mask_.expand_as(x)
            # f int(res) == self.aux.output_res:
            if return_feature is not None and int(res) in return_feature:
                feat_.append(x)
            i += 1
        img_ = img

        x = img = None
        for res, cur_ws in zip(self.synthesis.block_resolutions, block_ws):
            block = getattr(self.synthesis, f'b{res}')
            x, img = block(x, img, cur_ws, **kwargs)

        mask = common_utils.resize(t_mask, (img.shape[-2], img.shape[-1]))
        add_img = img_ * mask

        if return_feature is not None:
            return img_, img, add_img, mask, feat
        else:
            return img_, img, add_img, mask

    def synthesis_forward_slowly(self, w_samples, **synthesis_kwargs):
        return slowly_forward(self.synthesis_forward, w_samples, **synthesis_kwargs)

    def synthesis_aux_forward_slowly(self, w_samples, **synthesis_kwargs):
        return slowly_forward(self.synthesis_aux_forward, w_samples, **synthesis_kwargs)

    def forward(self, z, c = None, is_latent = False, aux = False, **synthesis_kwargs):
        if is_latent is True:
            w = z
        else:
            w = self.mapping(z, c)
        if aux is True:
            return self.synthesis_aux_forward(w, **synthesis_kwargs)[0]
        else:
            return self.synthesis_forward(w, **synthesis_kwargs)


def get_parameter(G, index = [4, 5, 6, 7, 8], **opt_kwargs):
    self = G
    params = []
    for i, res in enumerate(self.synthesis.block_resolutions):
        block = getattr(self.synthesis, f'b{res}')
        if i in index:
            params.append({'params': block.parameters(), **opt_kwargs})
    # params.append({'params': G.style_res, 'lr': 1e-4})
    return params


def generate_images_slowly(synthesis_forward, w_samples):
    num = len(w_samples)
    images = None
    for i in range(num):
        w = w_samples[[i], ...]
        image = synthesis_forward(w, noise_mode = 'const', force_fp32 = True)
        images = common_utils.torch_cat([images, image], dim = 0)
    return images


def lerp(w1, w2, beta):
    if not w1.shape == w2.shape:
        print(w1.shape, w2.shape)
        raise
    w = w1 * beta + w2 * (1 - beta)
    return w


def check_shape(a, b):
    for i, s in enumerate(a.shape):
        if s != b[i]:
            raise Exception('Shape error, input shape {}, correct shape {}'.format(a.shape, b))


def get_noise_bufs(G, mode = 'stylegan3'):
    if mode == 'stylegan2':
        noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    else:
        noise_bufs = []
    return noise_bufs

def laplacian(ws, normalize = True):
    if normalize:
        eye = torch.eye(len(ws)).to(ws.device)
        W = torch.exp(-(ws.unsqueeze(1) - ws.unsqueeze(0)).norm(dim = -1, p = 2) / 128) - eye
        W = torch.from_numpy(np.around(W.cpu().numpy(),decimals = 4)).to(W.device)
        sqrt_D = torch.diag(torch.sum(W, dim = -1).pow(-0.5))
        L = eye - sqrt_D @ W @ sqrt_D
    else:
        eye = torch.eye(len(ws)).to(ws.device)
        W = torch.exp(-(ws.unsqueeze(1) - ws.unsqueeze(0)).norm(dim = -1, p = 2) / 128) - eye
        W = torch.from_numpy(np.around(W.cpu().numpy(),decimals = 4)).to(W.device)
        D = torch.diag(torch.sum(W, dim = -1))
        L = D - W
    return L

class Slicing_torch(torch.nn.Module):
    def __init__(self, repeat_rate = 1, num_slice = 36, resize = None):
        super().__init__()
        # Number of directions
        self.num_slice = num_slice
        self.repeat_rate = repeat_rate
        self.resize = resize

    def update_slices(self, layers):
        directions = []
        for l in layers:  # converted to [B, W, H, D]
            if l.ndim == 4:
                if self.resize is not None:
                    l = common_utils.adaptive_pool(l, self.resize)
                l = l.permute(0, 2, 3, 1)
            dim_slices = l.shape[-1]
            num_slices = self.num_slice
            cur_dir = torch.randn(size = (num_slices, dim_slices)).to(l.device)
            norm = torch.sqrt(torch.sum(torch.square(cur_dir), axis = -1))
            norm = norm.view(num_slices, 1)
            cur_dir = cur_dir / norm
            directions.append(cur_dir)
        self.directions = directions
        self.target = self.compute_target(layers)

    def compute_proj(self, input, layer_idx, repeat_rate):
        if input.ndim == 4:
            if self.resize is not None:
                input = common_utils.adaptive_pool(input, self.resize)
            input = input.permute(0, 2, 3, 1)

        batch = input.size(0)
        dim = input.size(-1)
        tensor = input.view(batch, -1, dim)
        tensor_permute = tensor.permute(0, 2, 1)
        sliced = torch.matmul(self.directions[layer_idx], tensor_permute)

        # # Sort projections for each direction
        sliced, _ = torch.sort(sliced)
        sliced = sliced.repeat_interleave(repeat_rate ** 2, dim = -1)
        sliced = sliced.view(batch, -1)
        return sliced

    def compute_target(self, layers):
        target = []
        # target_sorted_sliced = []
        for idx, l in enumerate(layers):
            # target_sorted_sliced.append(l)
            sliced_l = self.compute_proj(l, idx, self.repeat_rate)
            target.append(sliced_l.detach())
        return target

    def forward(self, input):
        loss = 0.0
        # output = []
        for idx, l in enumerate(input):
            cur_l = self.compute_proj(l, idx, 1)
            loss += F.mse_loss(cur_l, self.target[idx].expand(cur_l.size(0), -1))
        loss /= len(input)
        return loss
