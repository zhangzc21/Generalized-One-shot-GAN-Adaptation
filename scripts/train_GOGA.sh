############################################
##############MUST BE DEFINED###############
image_path="test_data/references/000_245_709_4k_carlos-alberto-color-study_00.png" # path to the reference image
out_dir="Results" # dir to save results
mask_dir="test_data/masks" # all entity masks should be placed in the mask_dir
############################################
#############alternative###################
test_image_path="test_data/real_images"
############################################

python generalized_one_shot_adaption.py --exp_name  test \
--image_path $image_path \
--out_dir $out_dir \
--mask_dir $mask_dir \
--test_image_path $test_image_path \
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
--flip_aug True \
--e4e_model pretrained_models/e4e_ffhq_encode.pt
