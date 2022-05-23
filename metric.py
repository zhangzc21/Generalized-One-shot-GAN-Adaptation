import os
import argparse

import torch
from tqdm import tqdm
import pathlib

@torch.no_grad()
def calc_metric(G_source_paths, G_target_paths):
    resize256 = torch.nn.AdaptiveAvgPool2d((256, 256))
    pbar = tqdm(zip(G_source_paths, G_target_paths), total = len(G_target_paths))

    xs_landmarks = np.load(r'D:\One-shot-Adaption-out\metric_results\source_images_landmarks.npy')

    array_to_tensor = lambda x: ((torch.from_numpy(x) - 127.5) / 127.5).permute(2,0,1).unsqueeze(0)

    lmk_NMEs= []
    id_SIMs = []

    for i, (source_path, target_path) in tqdm(enumerate(pbar)):
        x = common_utils.read_image(source_path, type = 'array', range = '0,255')
        y = common_utils.read_image(target_path, type = 'array', range = '0,255')
        ### landmarks Normalized Mean Error ####
        x_landmarks = fa.get_landmarks_from_image(x)
        y_landmarks = fa.get_landmarks_from_image(y)

        if y_landmarks is not None and len(y_landmarks) > 0:
            lmk_NME = np.linalg.norm(x_landmarks - y_landmarks[0], ord = 2) / 1024
            lmk_NMEs.append(lmk_NME)
        else:
            lmk_NME = '100'
        ### ID similarity #####
        x = array_to_tensor(x).to(device).to(torch.float)
        y = array_to_tensor(y).to(device).to(torch.float)
        x = resize256(x)
        y = resize256(y)
        sim = idloss.similarity(x, y)
        id_SIMs.append(float(sim))
        pbar.set_postfix(id_sim = float(sim), lmk_NME = float(lmk_NME))
    return lmk_NMEs, id_SIMs


if __name__ == '__main__':
    import common_utils
    import numpy as np
    import pickle as pkl
    from models.arcface.id_loss import IDLoss
    import cv2
    import face_alignment #https://github.com/1adrianb/face-alignment

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', type = str, default = r'', help = 'reference image name')
    parser.add_argument('--model_name', type = str, default = r'', help = 'model name')

    parser.add_argument('--source_dir', type = str, default = r'', help = 'dir to store the generated source images')
    parser.add_argument('--target_dir', type = str, default = r'', help = 'dir to store the adapted images')
    args = parser.parse_args()

    image_path = args.image_path
    image_name = os.path.basename(image_path).split('.')[0]

    yaml_path = f'metric_results/quantitative/{image_name}/{args.model_name}.yaml'
    if os.path.exists(yaml_path):
        exit()

    ### change here to the dir of source and adapted syntheses ###
    source_domain = [f'{args.source_dir}/{k}.png' for k in range(1000)]
    target_domain = [f'{args.target_dir}/{k}.png' for k in range(1000)]
    #############################################################
    device = torch.device('cuda')
    idloss = IDLoss().to(device)
    idloss.requires_grad = False

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device = 'cuda', face_detector = 'sfd')

    img = cv2.imread(image_path)
    img = cv2.resize(img, (1024, 1024))
    img = (img[:, :, ::-1].transpose(2, 0, 1) / 255. - 0.5) / 0.5
    img = torch.from_numpy(img).float().unsqueeze(0).to(device)

    lmk_NMEs, id_SIMs= calc_metric(source_domain, target_domain)
    print('lmk_NMEs:', len(lmk_NMEs), np.mean(lmk_NMEs))
    print('Identity similarity:', np.mean(id_SIMs))

    results = dict(image_name = image_name, effective_num =len(lmk_NMEs), id_sim = float(np.mean(id_SIMs)), lmk_NMEs = float(np.mean(lmk_NMEs)))

    import yaml
    root = pathlib.Path(f'metric_results/quantitative/{image_name}')
    root.mkdir(exist_ok = True, parents = True)
    with open(yaml_path,
              'w') as file:
        yaml.dump(results, file)
