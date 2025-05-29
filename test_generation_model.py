# Import necessary libraries and packages
import os
import argparse
from data import BasicDataset,get_image_mask_pairs
import cv2
from unet_models import StyleUnetGenerator
from utils import set_requires_grad, mixed_list, noise_list, image_noise
import torch.utils.data as data
import transforms as transforms
import torch
import shutil
import numpy as np
import torch.nn as nn
import os
import random
from tqdm import tqdm
from utils import calculate_fid
from skimage.metrics import structural_similarity as ssim






import sys

# Simulate command-line arguments

sys.argv = ['test_generation_model.py', '--test_set_dir','D:/MSc AV Project/cGAN-Seg-main_2/cGAN-Seg_datasets/FW_real+gen_2/FW_strided_10_test/test',
            '--gen_ckpt_dir','D:/MSc AV Project/cGAN-Seg-main_2/Pretrain+Finetune_modified_FW_real25_fce_tv(0.3,0.7)_lrsched0.0001_2/Gen.pth',
            '--output_dir','D:/MSc AV Project/cGAN-Seg-main_2/Pretrain+Finetune_modified_FW_real25_fce_tv(0.3,0.7)_lrsched0.0001_2/Gen']





# Setting the seed for generating random numbers for reproducibility
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def normalize_image(image):
    # Normalize the image to a range of [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val + 1e-8)  # Adding a small constant to avoid division by zero


def save_images(output_dir, real_img_list, fake_img_list, real_A_list):
    for i, (real_img, fake_img, real_A) in enumerate(zip(real_img_list, fake_img_list, real_A_list)):
        real_img_save = (real_img * 255).astype(np.uint8)
        fake_img_save = (fake_img * 255).astype(np.uint8)
        real_A_save = (real_A * 255).astype(np.uint8)
        # Save images
        cv2.imwrite(os.path.join(output_dir, 'real_images', f'images_{i:04d}.png'), real_img_save)
        cv2.imwrite(os.path.join(output_dir, 'gen_images', f'images_{i:04d}.png'), fake_img_save)
        cv2.imwrite(os.path.join(output_dir, 'real_masks', f'images_{i:04d}.png'), real_A_save)

def reshape_and_repeat(images_list, image_size, is_gray=True):
    # Reshape images for FID calculation
    num_images = len(images_list)
    #print(num_images)

    if is_gray:
        images = np.array(images_list).reshape(num_images, 1, image_size[0], image_size[1])
        return np.repeat(images, 3, axis=1)  # Repeat grayscale channel 3 times
    else:
        return np.array(images_list).reshape(num_images, 3, image_size[0],image_size[1])  # Already RGB, no need to repeat

def calculate_ssim(real_images, generated_images):
    """
    Calculate SSIM between real and generated images
    Args:
        real_images: List of real images (normalized 0-1)
        generated_images: List of generated images (normalized 0-1)
    Returns:
        Average SSIM score
    """
    ssim_scores = []
    for real, gen in zip(real_images, generated_images):
        # SSIM works on single channel images
        score = ssim(real, gen, 
                    data_range=1.0,  # Since images are normalized 0-1
                    win_size=11,     # Standard window size
                    channel_axis=None)  # For single channel images
        ssim_scores.append(score)
    return np.mean(ssim_scores)

def test(args, image_size=[512, 768], image_means=[0.5], image_stds=[0.5], batch_size=1):
    # Using CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Applying transformations on the test data
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])
    # Read samples
    sample_pairs = get_image_mask_pairs(args.test_set_dir)
    assert len(sample_pairs) > 0, f'No samples found in {args.test_set_dir}'
    #print(sample_pairs)

    # Load the test dataset and apply the transformations
    test_data = BasicDataset(sample_pairs, transforms=test_transforms)

    # Create a dataloader for the test dataset
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Load the models
    Gen = StyleUnetGenerator(style_latent_dim = 128,output_nc=1).to(device)
    Gen.load_state_dict(torch.load(args.gen_ckpt_dir,weights_only=True))

    # Set the models to evaluation mode
    Gen.eval()

    real_img_list=[]
    real_A_list=[]
    fake_img_list=[]
    set_requires_grad(Gen, False)
    with torch.no_grad():
        # Iterating over batches of test data
        for step, batch in enumerate(tqdm(test_iterator)):
            real_img = batch['image'].to(device=device, dtype=torch.float32)
            real_mask = batch['mask'].to(device=device, dtype=torch.float32)

            style = mixed_list(real_img.shape[0], 5, 128, device=device) if random.random() < 0.9 else noise_list(real_img.shape[0], 5, 128, device=device)  # latent_dim set to 128
            im_noise = image_noise(real_mask.shape[0], image_size, device=device)

            fake_img = Gen(real_mask, style, im_noise)
            fake_img = normalize_image(fake_img.cpu().numpy()[0, 0, :, :])
            real_img = normalize_image(real_img.cpu().numpy()[0, 0, :, :])

            real_mask = normalize_image(real_mask.cpu().numpy()[0, 0, :, :])

            real_img_list.append(real_img)
            real_A_list.append(real_mask)
            fake_img_list.append(fake_img)

    if os.path.exists(os.path.join(args.test_set_dir, 'images')):
        # Saving the real and generated images
        save_images(args.output_dir, real_img_list, fake_img_list, real_A_list)

        # Reshape and repeat images for FID calculation
        real_images = reshape_and_repeat(real_img_list, image_size, is_gray=True)
        generated_images = reshape_and_repeat(fake_img_list, image_size, is_gray=True)

        # Calculate SSIM (using original 0-1 normalized images)
        ssim_score = calculate_ssim(real_img_list, fake_img_list)
        
        # Calculate and print FID score
        fid_score = calculate_fid(real_images, generated_images, device)
        print('FID Score: %f' % fid_score)
        print(f'SSIM Score: {ssim_score:.4f}')
        
        # Save metrics to file
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
            f.write(f'FID Score: {fid_score:.4f}\n')
            f.write(f'SSIM Score: {ssim_score:.4f}\n')


if __name__ == "__main__":
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set_dir", required=True, type=str, help="path for the test dataset")
    ap.add_argument("--gen_ckpt_dir", required=True, type=str, help="path for the generator model checkpoint")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()

    # Check if test dataset directory exists
    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir

    if os.path.exists(args.output_dir):
        # Remove the existing directory and all its contents
        shutil.rmtree(args.output_dir)

    # Create the new directories
    os.makedirs(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, 'real_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'gen_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'real_masks'), exist_ok=True)
    # Call the test function
    test(args)
