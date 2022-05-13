test_image='D:/NIPS_2022/test_data'
out_dir='1'
adapted_model='adapted_models/rena.pkl'

python inference.py \
--test_image $test_image \
--out_dir $out_dir \
--adapted_model $adapted_model \
--pretrained_model pretrained_models/ffhq.pkl \
--e4e_model pretrained_models/e4e_ffhq_encode.pt