import os
import pickle as pkl
from latent_projector import ensemble_projector
from inversion_networks_utils import *
import warnings

warnings.filterwarnings("ignore")



class Inference():
    def __init__(self, pretrained_model, adapted_model, **kwargs):
        self.device = torch.device('cuda')
        self.load_source_model(pretrained_model)
        self.load_target_model(adapted_model)
        self.projector = None
    def load_source_model(self, model_path):
        with open(model_path, 'rb') as f:
            file = pkl.load(f)
            self.G = file['G_ema'].to(self.device)
            self.D = file['D'].to(self.device)

    def load_target_model(self, model_path):
        if not os.path.exists(model_path):
            print(f'No pre-trained model')
        else:
            pkl = torch.load(model_path)
            cur_step = pkl['step']
            self.G_target = pkl['G_target']
            self.use_aux = pkl['aux']
            print(f'Loading pretrained model from step {cur_step}')
    @torch.no_grad()
    def random_synthesis(self, num = 2, aux = False, out_dir = None, show = False):
        ws, _ = self.generate_test_latent(w=None,sample_num = num)
        adapted_images = slowly_forward(self.G_target, ws, aux = aux, is_latent = True, noise_mode = 'const', force_fp32 = True)
        source_images = slowly_forward(self.G.synthesis, ws, noise_mode = 'const', force_fp32 = True)
        names = list(range(len(ws)))
        for w, source_image, stylized_image, name in zip(ws, source_images, adapted_images, names):
            image = torch.cat([source_image.unsqueeze(0), stylized_image.unsqueeze(0)], dim = 0)
            if out_dir is not None:
                common_utils.save_image(image, os.path.join(config.out_dir, name + f'.jpg'),
                                        nrow = 1, range = '-1,1')
            if show is True:
                common_utils.show_tensor_image(image, nrow = 2)

    @torch.no_grad()
    def transfer(self, image_path, aux = False, out_dir = None, show = False):
        ws, images, masks, names = self.dataloader(image_path, flip_aug = False, use_mask = False, return_name = True, e4e_model = config.e4e_model)
        adapted_images = slowly_forward(self.G_target, ws, aux = aux, is_latent = True, noise_mode = 'const',
                                        force_fp32 = True)
        source_images = slowly_forward(self.G.synthesis, ws, noise_mode = 'const', force_fp32 = True)
        for w, image, source_image, stylized_image, name in zip(ws, images, source_images, adapted_images, names):
            image = torch.cat([image, source_image.unsqueeze(0), stylized_image.unsqueeze(0)], dim = 0)
            if out_dir is not None:
                common_utils.save_image(image, os.path.join(config.out_dir, name + f'.jpg'),
                                        nrow = 1, range = '-1,1')
            if show is True:
                common_utils.show_tensor_image(image, nrow = 3)

    def dataloader(self, image_paths, flip_aug = False,use_mask = False, return_name = False, e4e_model = None):
        ws = []
        images = []
        masks = []
        names = []
        common_utils.make_dir('inversion_out/latent_code')  # latent code dir

        image_paths = common_utils.load_image_paths(image_paths)
        assert isinstance(image_paths, list)

        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image = common_utils.read_image(image_path).to(self.device)
            image = common_utils.resize(image, (self.G.img_resolution, self.G.img_resolution))
            save_path = f'inversion_out/latent_code/{image_name}_w.pkl'
            label_path = os.path.join('data\paper_attribute_mask',
                                      os.path.basename(image_path).split('.')[0]+ '_mask.png')
            if not os.path.exists(label_path):
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

            if os.path.exists(save_path):
                w = torch.load(save_path)
            else:
                if self.projector is None:
                    self.projector = ensemble_projector(e4e_model = e4e_model)
                    self.projector.G = self.G
                inv_img, w = self.projector.calc_inversions(image, mask =mask_label)
                torch.save(w, save_path)
                common_utils.save_image(torch.cat([inv_img, image], dim = -1), f'inversion_out/latent_code/{image_name}.jpg')
            images.append(image)
            ws.append(w)
            names.append(image_name)
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
                    inv_img_flip, w_flip = self.projector.calc_inversions(image_flip, mask =torch.flip(mask_label, dims = [-1]))
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
            # for i in range(linspace_num, 0, -1):
            #     if w is None:
            #         w = w_samples[[0]]
            #     w_lerp = lerp(w.repeat(linspace_num, 1, 1), w_samples[:5], i / linspace_num)
            #     ws_lerp.append(w_lerp)
            # ws_lerp = rearrange(ws_lerp, 'a b c d -> (b a) c d', a = len(ws_lerp), b = w_lerp.shape[0],
            #                     c = w_lerp.shape[1], d = w_lerp.shape[2])
        return w_samples, ws_lerp
        ### init test latent code ####################################################################

    @torch.no_grad()
    def semantic_manipulation(self, w, name = ''):
        from latent_editor_wrapper import LatentEditorWrapper
        latent_editor = LatentEditorWrapper()
        latents_after_edit = latent_editor.get_single_interface_gan_edits(w, [-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0])
        synthesis_kwargs = dict(noise_mode = 'const', force_fp32 = True, mix_style = True)
        for direction, factor_and_edit in latents_after_edit.items():
            # print(f'Showing {direction} change')
            i = 0
            for factor, latent in factor_and_edit.items():
                old_image = self.G.synthesis(latent, **dict(noise_mode = 'const', force_fp32 = True))
                new_image_full,new_image,_,_ = self.G_target.synthesis_aux_forward(latent, alpha = 0.0, **synthesis_kwargs)
                # images = torch.cat([, new_image,new_image_full], dim = 0)
                common_utils.save_image(old_image, config.out_dir + f'/manipulation/{name}_{direction}_{i}_source.jpg')
                common_utils.save_image(new_image, config.out_dir + f'/manipulation/{name}_{direction}_{i}_new.jpg')
                common_utils.save_image(new_image_full, config.out_dir + f'/manipulation/{name}_{direction}_{i}_full.jpg')
                i += 1


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--test_image', default = r'data/style_images_aligned_2/arcane_jayce.png',
                        type = str)
    parser.add_argument('--out_dir', type = str, default = None)
    parser.add_argument('--adapted_model', type = str, default = r'pretrained_models\ffhq.pkl',
                        help = 'path of stylegan pkl file')
    parser.add_argument('--pretrained_model', type = str, default = r'pretrained_models\ffhq.pkl',
                        help = 'path of stylegan pkl file')
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--e4e_model', type = str, default = 'pretrained_models/e4e_ffhq_encode.pt')
    parser.add_argument('--use_aux', type = bool, default = False)
    opt = parser.parse_args()
    config = vars(opt)
    config = common_utils.EasyDict(**config)
    config.synthesis_kwargs = dict(noise_mode = 'const', force_fp32 = True, mix_style = True)
    config.mapping_kwargs = dict(c = None, truncation_psi = 0.7)
    inference = Inference(**config)
    inference.random_synthesis(num = 1, aux = inference.use_aux, show = True)
    inference.transfer(config.test_image, aux = inference.use_aux, show = True)