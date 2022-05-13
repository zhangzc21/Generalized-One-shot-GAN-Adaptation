test_image='' # path to the real image that will be inverted by e4e
out_dir=''  # save dir
adapted_model='' # path to adapted model

python inference.py \
--test_image $test_image \
--out_dir $out_dir \
--adapted_model $adapted_model \
--pretrained_model pretrained_models/ffhq.pkl \
--e4e_model pretrained_models/e4e_ffhq_encode.pt