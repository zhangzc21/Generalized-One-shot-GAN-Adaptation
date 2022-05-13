image_path="" # path to the reference image

python generalized_one_shot_adaption.py --exp_name  test \
--pretrained_model "pretrained_models/ffhq.pkl" \
--source_domain face \
--index 8 \
--total_step 2000 \
--batch 1 \
--fix_style True \
--lpips_weight 10 \
--entity_weight 2 \
--reg_weight 1 \
--style_weight 0.2 \
--vgg_feature_num 2 \
--use_mask True \
--image_path $image_path \
--flip_aug True \
--e4e_model pretrained_models/e4e_ffhq_encode.pt
