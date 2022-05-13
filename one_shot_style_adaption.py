from PIL import Image
import os
from PIL import Image
import pickle as pkl
from tqdm import tqdm
from latent_projector import ensemble_projector
import cv2
import lpips
import copy
from training.loss import CLIPLoss
from einops import rearrange
from inversion_networks_utils import *
from SSIM import MS_SSIM
from termcolor import cprint
import warnings

warnings.filterwarnings("ignore")


class DomainAdaption():
    def __init__(self, pretrained_model):
        self.device = torch.device('cuda')
        self.load_source_model(pretrained_model)
        self.projector = None

    def read_image(self, img_path, img2tensor = True):
        image = np.ascontiguousarray(cv2.imread(img_path)[..., ::-1])
        image = image / 127.5 - 1
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        self.image_name = os.path.basename(img_path)
        return image

    def load_source_model(self, model_path):
        with open(model_path, 'rb') as f:
            file = pkl.load(f)
            self.G = file['G_ema'].to(self.device)
            self.D = file['D'].to(self.device)

    def load_target_model(self, model_path):
        if os.path.exists(model_path):
            pretrained_path = config.out_dir + '/G_target.pkl'
        else:
            print(f'No pre-trained model')

        if pretrained_path is not None:
            pkl = torch.load(pretrained_path)
            cur_step = pkl['step']
            self.G_target = pkl['G_target']
            print(f'Loading pretrained model from step {cur_step}')

    def dataloader(self, image_paths, flip_aug = False, use_mask =False, return_name = False, e4e_model = None):
        ws = []
        images = []
        masks = []
        names = []
        common_utils.make_dir('inversion_out/latent_code')  # latent code dir

        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                import PIL
                _all_fnames = [os.path.join(root, fname) for root, _dirs, files in os.walk(image_paths) for fname in
                               files]
                PIL.Image.init()
                image_paths = sorted(
                    fname for fname in _all_fnames if os.path.splitext(fname)[1].lower() in PIL.Image.EXTENSION)
            elif os.path.exists(image_paths):
                image_paths = [image_paths]

        assert isinstance(image_paths, list)

        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image = self.read_image(image_path)
            image = common_utils.resize(image, (self.G.img_resolution, self.G.img_resolution))
            save_path = f'inversion_out/latent_code/{image_name}_w.pkl'
            label_path = os.path.join('data\style_images_aligned_mask',
                                      os.path.basename(image_path).replace('.png', '_mask.png'))
            if os.path.exists(save_path):
                w = torch.load(save_path)
            else:
                if self.projector is None:
                    self.projector = ensemble_projector(e4e_model = e4e_model)
                    self.projector.G = self.G
                inv_img, w = self.projector.calc_inversions(image)
                torch.save(w, save_path)
                common_utils.save_image(torch.cat([inv_img, image], dim = -1), f'inversion_out/latent_code/{image_name}.jpg')
            images.append(image)
            ws.append(w)
            names.append(image_name)

            if not os.path.exists(label_path) or not use_mask:
                mask_label = torch.zeros_like(image)[:, [0], ...]
                print('no pre-defined mask, using zeros mask')
            else:
                print(f'use pre-defined mask: {label_path}')
                mask_label = common_utils.read_image(label_path).to(image.device)
                mask_label = (mask_label + 1) / 2
                mask_label = mask_label[:, [0], ...]
                mask_label = common_utils.resize(mask_label, (self.G.img_resolution, self.G.img_resolution))
                mask_label[mask_label > 0.5] = 1
                mask_label[mask_label <= 0.5] = 0
            masks.append(mask_label)

            if flip_aug:
                save_path = f'inversion_out/latent_code/{image_name}_flip_w.pkl'
                image_flip = torch.flip(image, dims = [-1])
                if os.path.exists(save_path):
                    w_flip = torch.load(save_path)
                else:
                    if self.projector is None:
                        self.projector = ensemble_projector(e4e_model = e4e_model)
                        self.projector.G = self.G
                    inv_img_flip, w_flip = self.projector.calc_inversions(image_flip)
                    torch.save(w_flip, save_path)
                    common_utils.save_image(torch.cat([inv_img_flip, image_flip], dim = -1),
                                     f'inversion_out/latent_code/{image_name}_flip.jpg')
                images.append(image_flip)
                ws.append(w_flip)
                masks.append(torch.flip(mask_label, dims = [-1]))
                names.append(image_name + '_flip')
        if return_name:
            return ws, images, masks, names
        else:
            return ws, images, masks

    def generate_test_latent(self, w, sample_num = 36, linspace_num = 5):
        ### init test latent code ########################

        with torch.no_grad():
            z_samples = np.random.RandomState(123).randn(sample_num, 512)
            w_samples = self.G.mapping(torch.from_numpy(z_samples).to(config.device), **config.mapping_kwargs)
            ws_lerp = []

            for i in range(linspace_num, 0, -1):
                if w is None:
                    w = w_samples[[0]]
                w_lerp = lerp(w.repeat(linspace_num, 1, 1), w_samples[:5], i / linspace_num)
                ws_lerp.append(w_lerp)
            ws_lerp = rearrange(ws_lerp, 'a b c d -> (b a) c d', a = len(ws_lerp), b = w_lerp.shape[0],
                                c = w_lerp.shape[1], d = w_lerp.shape[2])
        return w_samples, ws_lerp
        ### init test latent code ####################################################################

    def process(self, ws, images, masks, config):
        ### BEGIN: parameter ##############
        device = config.device
        total_step = config.total_step
        zero_tensor = torch.tensor([0.0]).to(device)
        self.w = ws
        sample_num = 16
        linspace_num = 5
        w_ref = torch.mean(ws, dim = 0, keepdim = True)
        ### END: parameter ##################################

        ### BEGIN: Define Generators ##############
        self.G_source = MaskStyleGAN(copy.deepcopy(self.G), w_ref, index = config.index).to(device)
        self.G_target = MaskStyleGAN(copy.deepcopy(self.G), w_ref, index = config.index).to(device)
        # self.D = Discriminator(self.D)
        w_samples, ws_lerp = self.generate_test_latent(w_ref, sample_num, linspace_num)
        cur_step = 0
        if os.path.exists(config.out_dir + '/G_target.pkl'):
            pretrained_path = config.out_dir + '/G_target.pkl'
        else:
            pretrained_path = None

        if pretrained_path is not None:
            pkl = torch.load(pretrained_path)
            cur_step = pkl['step']
            self.G_target = pkl['G_target']
            print(f'Loading pretrained model from step {cur_step}')

            if cur_step >= total_step:
                return

        common_utils.toggle_grad([self.G_target.mapping, self.G_target.synthesis, self.G_target.aux, self.D], flag = True)
        optimizer_G = torch.optim.Adam(get_parameter(self.G_target, list(range(0, 9)), **config.Gopt_kwargs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max = total_step, eta_min = 1e-4)
        ### END: Define Generators ############################################################

        ### BEGIN: Read image #################
        with torch.no_grad():
            real_contents = self.G_source.synthesis_forward(ws, **config.synthesis_kwargs)
            real_styles = images
            mask_labels = masks
        ### END: Read image ###########################################################################

        ### BEGIN: Define Loss ##########
        self.clip_loss = CLIPLoss(device = device)
        self.ssim_loss = MS_SSIM(window_size = 11, window_sigma = 1.5, data_range = 1.0, channel = 3).to(device)
        self.lpips_loss = lpips.LPIPS(net = 'vgg').to(device).eval()
        with torch.no_grad():
            real_style_lpips_feats = self.lpips_loss.get_feature(common_utils.downsample(real_styles, (256, 256)))[
                                     -config.vgg_feature_num:]
            real_content_lpips_feats = self.lpips_loss.get_feature(common_utils.downsample(real_contents, (256, 256)))[
                                       -config.vgg_feature_num:]
        self.color_loss = Slicing_torch(num_slice = 256).to(device)
        ### END: Define Loss ###########################################################################

        ### BEGIN: Training ##################################
        if config.use_wandb:
            import wandb
            wandb.init(project = 'stylegan3', name = os.path.basename(config.image_path).split('.')[0]+ '-'+ config.exp_name)
        print('View results at:')
        cprint('                {}'.format(os.path.join(os.getcwd(), config.out_dir)), color = 'cyan')
        pbar = tqdm(range(cur_step, total_step), total = total_step, initial = cur_step)
        kernel = torch.ones(3, 3).to(device)
        for step_idx in pbar:
            rand = torch.randint(0, len(ws), [1]).tolist()
            w = ws[rand, ...]
            real_content = real_contents[rand, ...]
            real_style = real_styles[rand, ...]
            mask_label = mask_labels[rand, ...]
            real_style_lpips_feat = [f[rand, ...] for f in real_style_lpips_feats]
            self.color_loss.update_slices(real_style_lpips_feat)

            ### begin optimize generator ###
            optimizer_G.zero_grad()
            with torch.no_grad():
                z = torch.randn(config.batch, 512).to(device)
                w_ = self.G_target.mapping(z, **config.mapping_kwargs)
                lap = laplacian(torch.cat([w, w_], dim = 0)[:, :config.index, :].flatten(-2), normalize = False)
                synth_content = self.G_source.synthesis_forward(w_, **config.synthesis_kwargs)

            synth_style = self.G_target.synthesis_forward(
                w_, **config.synthesis_kwargs)  # gan lpips_loss #target feature + 1
            syn_style_lpips_feat = self.lpips_loss.get_feature(common_utils.adaptive_pool(synth_style, (256, 256)))[
                                   -config.vgg_feature_num:]

            ### aug ##
            rec_style = self.G_target.synthesis_forward(w, **config.synthesis_kwargs)
            ### aug ##############
            rec_style_lpips_feat = self.lpips_loss.get_feature(common_utils.adaptive_pool(rec_style, (256, 256)))[
                                   -config.vgg_feature_num:]
            rec_style_lpips_feat = [f.detach() for f in rec_style_lpips_feat]
            self.color_loss.update_slices(rec_style_lpips_feat)
            color_loss = self.color_loss(list(syn_style_lpips_feat))
            lapReg = self.clip_loss.VLapR(real_content, synth_content, real_style,
                                          synth_style, lap)
            ssim_loss = (1 - self.ssim_loss(common_utils.denorm(rec_style), common_utils.denorm(real_style))).mean()
            lpips_loss = self.lpips_loss(common_utils.resize(rec_style, (256, 256)),
                                         common_utils.resize(real_style, (256, 256))).mean()
            # dis_rec_loss = sum([F.l1_loss(a, b) for a, b in zip(real_style_dis_feat, rec_style_full_dis_feat)])/len(rec_style_full_dis_feat)
            loss = config.lpips_weight * (
                    lpips_loss + ssim_loss) + config.style_weight * color_loss + config.reg_weight * lapReg
            loss.backward()
            if step_idx > 0:
                optimizer_G.step()
            scheduler.step(epoch = step_idx)
            ### end optimize generator ###

            ### Log ###
            loss_dict = dict(lpips_loss = lpips_loss,
                             errG_color = color_loss,
                             lapReg = lapReg,
                             ssim_loss = ssim_loss)
            for key in loss_dict.keys():
                loss_dict[key] = float(loss_dict[key])
            pbar.set_postfix(**loss_dict)
            if config.use_wandb:
                wandb.log(loss_dict)

            ### Save image ###
            with torch.no_grad():
                if (step_idx + 1) % 200 == 0 or step_idx == 0:
                    torch.save({'step': step_idx + 1, 'G_target': self.G_target, 'aux': False}, config.out_dir + f'/G_target.pkl')
                    common_utils.save_image(
                        torch.cat(
                            [rec_style,
                             real_style],
                            dim = 0),
                        path = config.out_dir + f'/rec_{step_idx + 1}.jpg', nrow = 2, range = '-1,1')

                    style_images_lerp = slowly_forward(self.G_target.synthesis_forward, w_samples, **config.synthesis_kwargs)
                    common_utils.save_image(style_images_lerp, config.out_dir + f'/random_{step_idx + 1}.jpg',
                                            nrow = int(np.sqrt(sample_num)), pad_value = 1.0, padding = 10)
        ### Save image ###
        with torch.no_grad():
            style_images_lerp = generate_images_slowly(
                functools.partial(self.G_target.synthesis_forward, mix_style = False),
                w_samples)
            common_utils.save_image(style_images_lerp, config.out_dir + f'/random_target.jpg',
                                    nrow = int(np.sqrt(sample_num)), pad_value = 1.0, padding = 10)

    @torch.no_grad()
    def semantic_manipulation(self, w, name = ''):
        from latent_editor_wrapper import LatentEditorWrapper
        latent_editor = LatentEditorWrapper()
        latents_after_edit = latent_editor.get_single_interface_gan_edits(w, [-2.0, 2.0])
        synthesis_kwargs = dict(noise_mode = 'const', force_fp32 = True, mix_style = True)
        for direction, factor_and_edit in latents_after_edit.items():
            # print(f'Showing {direction} change')
            i = 0
            for factor, latent in factor_and_edit.items():
                old_image = self.G.synthesis(latent, **dict(noise_mode = 'const', force_fp32 = True))
                new_image = self.G_target.synthesis_forward(latent, **synthesis_kwargs)
                images = torch.cat([old_image, new_image], dim = 0)
                common_utils.save_image(images, config.out_dir + f'/manipulation/{name}_{direction}_{i}.jpg')
                i += 1


import matplotlib.pyplot as plt


def plot_syn_images(syn_images):
    import pathlib
    fig_dir = pathlib.Path('fig')
    fig_dir.mkdir(exist_ok = True)
    import time
    for i, img in enumerate(syn_images):
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
        plt.axis('off')
        resized_image = Image.fromarray(img, mode = 'RGB').resize((256, 256))
        plt.imshow(resized_image)
        plt.show()
        t = time.time()
        ts = time.strftime('%Y-%m-%d %H-%M-%s', time.localtime(t))
        plt.savefig(f'{fig_dir}/{ts}.png')
        del img
        del resized_image
        torch.cuda.empty_cache()


def train_function(config):
    if isinstance(config.image_path, str):
        out_dir = os.path.join(config.out_dir, os.path.basename(config.image_path).split('.')[0], config.exp_name)
    else:
        from datetime import datetime
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
        out_dir = os.path.join(config.out_dir, date_time, config.exp_name)
    config.out_dir = out_dir
    common_utils.make_dir(config.out_dir)

    import yaml
    with open(os.path.join(config.out_dir, 'config.yaml'), 'w') as file:
        yaml.dump(dict(**config), file)

    domain_adaption = DomainAdaption(config.pretrained_model)
    ws, images, masks = domain_adaption.dataloader(config.image_path, flip_aug = config.flip_aug,
                                                   use_mask = config.use_mask, e4e_model = config.e4e_model)
    ws = torch.cat(ws, dim = 0)
    images = torch.cat(images, dim = 0)
    masks = torch.cat(masks, dim = 0)
    domain_adaption.process(ws, images, masks, config)


@torch.no_grad()
def test_function(config):
    domain_adaption = DomainAdaption(config.pretrained_model)
    domain_adaption.load_target_model(config.out_dir + '/G_target.pkl')
    ws, images, masks, names = domain_adaption.dataloader(config.test_image_path, flip_aug = True,
                                                          use_mask = config.use_mask, return_name = True,
                                                          e4e_model = config.e4e_model)
    ws = torch.cat(ws, dim = 0)
    images = torch.cat(images, dim = 0)
    # ws, _ = domain_adaption.generate_test_latent(domain_adaption.G.mapping.w_avg.repeat(1,domain_adaption.G.num_ws,1), sample_num = 16, linspace_num = 5)
    # names = [str(i) for i in range(len(ws))]
    source_images = slowly_forward(domain_adaption.G.synthesis, ws, noise_mode = 'const', force_fp32 = True)
    stylized_images = slowly_forward(domain_adaption.G_target.synthesis_forward, ws, **config.synthesis_kwargs)
    for w, source_image, stylized_image, name in zip(ws, source_images, stylized_images, names):
        common_utils.save_image(source_image.unsqueeze(0), os.path.join(config.out_dir, 'test', name + '.jpg'),
                                nrow = 1, range = '-1,1')
        common_utils.save_image(stylized_image.unsqueeze(0), os.path.join(config.out_dir, 'test', name + '_stylized.jpg'),
                                nrow = 1, range = '-1,1')
        # domain_adaption.semantic_manipulation(w.unsqueeze(0), name)


if __name__ == '__main__':
    from argparse import ArgumentParser
    source_G = {'face1024': 'F:/GITHUB/stylegan3/pretrained_models/ffhq.pkl',
                'cat512': r'D:\github\stylegan3\pretrained\stylegan2-afhqcat-512x512.pkl',
                'dog512': r'D:\github\stylegan3\pretrained\stylegan2-afhqdog-512x512.pkl',
                'church256': r'pretrained_models/stylegan2-church-config-f.pkl'}
    parser = ArgumentParser()

    parser.add_argument('--image_path', default = r'data/style_images_aligned_2/arcane_jayce.png',
                        type = str)
    parser.add_argument('--test_image_path', default = r'D:\NIPS_2022\faces_of_celebrities_aligned',
                        type = str)
    parser.add_argument('--out_dir', type = str, default = r'D:\One-shot-Adaption-out')
    parser.add_argument('--pretrained_model', type = str, default = r'F:\GITHUB\stylegan3\pretrained_models\ffhq.pkl',
                        help = 'path of stylegan pkl file')
    parser.add_argument('--total_step', type = int, default = 600, help = 'total optimization step')
    parser.add_argument('--exp_name', type = str, default = 'test')
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--batch', type = int, default = 1)

    parser.add_argument('--lpips_weight', type = float, default = 1, help = 'weight of lpips')
    parser.add_argument('--reg_weight', type = float, default = 1, help = 'weight of regularization')
    parser.add_argument('--style_weight', type = float, default = 3)
    parser.add_argument('--dis_weight', type = int, default = 0)
    parser.add_argument('--source_domain', type = str, default = 'face')

    parser.add_argument('--flip_aug', type = bool, default = False)
    parser.add_argument('--use_mask', type = bool, default = False)
    parser.add_argument('--fix_style', type = bool, default = False)
    parser.add_argument('--vgg_feature_num', type = int, default = 2)
    parser.add_argument('--index', type = int, default = 8)
    parser.add_argument('--e4e_model', type= str, default = None)
    parser.add_argument('--use_wandb', type = bool, default = False)

    opt = parser.parse_args()
    config = vars(opt)
    config = common_utils.EasyDict(**config)
    config.Gopt_kwargs = dict(lr = 1e-3, betas = (0, 0.999))
    config.synthesis_kwargs = dict(noise_mode = 'const', force_fp32 = True, mix_style = config.fix_style)
    config.mapping_kwargs = dict(c = None, truncation_psi = 0.7)
    train_function(config)
    test_function(config)
