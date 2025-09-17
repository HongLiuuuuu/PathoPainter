# extract_feat_256.py
import os
import argparse
import numpy as np
from PIL import Image
import torch
from hipt_model_utils import get_vit256

def extract_feat_256(mask_path, model256, device256):
    # Load and normalize mask
    mask = Image.open(mask_path)
    mask = np.array(mask)[None, None, ...] / 255.0
    mask = torch.tensor(mask).float().to(device256)

    # Load image
    img_path = mask_path.replace('mask', 'image')
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)[None, ...]
    img = torch.tensor(img).float().to(device256)

    # Extract features
    feat_256 = model256(img, mask)
    return feat_256


def main(args):
    # Set device
    device256 = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load model
    model256 = get_vit256(pretrained_weights=args.model_path).to(device256)
    model256.eval()

    # Read mask file list
    with open(args.mask_list, 'r') as f:
        mask_paths = [line.strip().split(', ')[0] for line in f]

    # Process each image
    for mask_path in mask_paths:
        feat_256 = extract_feat_256(mask_path, model256, device256)
        feat_256 = feat_256.squeeze(0).cpu().detach().numpy()

        # Save features
        save_path = mask_path.replace('mask', args.output_folder)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path.replace('.png', '.npy'), feat_256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ViT features from 256x256 images with mask.')
    parser.add_argument('--mask_list', type=str, required=True, help='Path to .txt file containing list of mask paths.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained ViT checkpoint.')
    parser.add_argument('--output_folder', type=str, default='ssl_features_test', help='Subdirectory to save output features (replaces "mask" in path).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index (default: 0)')
    args = parser.parse_args()

    main(args)
