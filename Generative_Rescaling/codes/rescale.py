import os
import argparse
import cv2
import torch
import yaml
from models import create_model
import options.options as option

def rescale_images(opt, input_dir, output_dir, scale, operation, gaussian_scale=1):
    model = create_model(opt)
    model.load()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            image = cv2.imread(input_path)
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(model.device)

            if operation == 'downscale':
                output_image_tensor = model.downscale(image_tensor)
            elif operation == 'upscale':
                output_image_tensor = model.upscale(image_tensor, scale, gaussian_scale)
            else:
                raise ValueError("Operation must be 'downscale' or 'upscale'")

            output_image = output_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, output_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rescale images.')
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    parser.add_argument('--operation', type=str, required=True, choices=['upscale', 'downscale'], help='Operation to perform: upscale or downscale')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    input_dir = opt['datasets']['test_1']['dataroot_GT']
    output_dir = input_dir + '_rescaled'
    scale = opt['scale']

    rescale_images(opt, input_dir, output_dir, scale, args.operation)
    print("Rescaling complete. Your images are saved in ", output_dir)
