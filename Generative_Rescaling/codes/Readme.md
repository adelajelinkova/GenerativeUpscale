# Train split model
python train_separately.py -opt options/train/my_train_NewIRN_x4.yml

# Test split model
python downscale_upscale_analyse.py -opt options/test/test_NewIRN_x4.yml

# Use rescaler
python rescale.py -opt options/test/test_NewIRN_x4.yml --operation upscale