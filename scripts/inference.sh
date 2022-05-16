############################################
##############MUST BE DEFINED###############
adapted_model='adapted_models/disney.pkl' # path to adapted model
############################################
##############alternative###################
test_image='test_data/real_images' # path or dir to the real images that will be inverted by e4e
out_dir='None'  # save dir
############################################

python inference.py \
--test_image $test_image \
--out_dir $out_dir \
--adapted_model $adapted_model \
--pretrained_model pretrained_models/ffhq.pkl \
--e4e_model pretrained_models/e4e_ffhq_encode.pt \
--show True