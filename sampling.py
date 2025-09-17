import torch
from pathlib import Path
from utils import get_model
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from PIL import Image
import os

# Set device
device = torch.device("cuda:0")

# Sampling parameters
index = 0
batch_size = 100  # Set the batch size to a higher value if GPU memory allows
scale = 2
ddim_steps = 50
shape = [3, 64, 64]


# Load pretrained weights and model
model_path = Path("logs/08-07T14-37_pathopainter/")
model, config = get_model(model_path, device, "last.ckpt")
sampler = PLMSSampler(model)

# Load dataset and prepare DataLoader
data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()
data.batch_size = batch_size

# Update dataset properties
data.datasets['train'].p_uncond = 0
data.datasets['train'].aug = 0

# Create a DataLoader for the training set
train_loader = DataLoader(data.datasets['train'], batch_size=batch_size, shuffle=False, num_workers=4)

# Inference with no gradient tracking
with torch.no_grad(), model.ema_scope():
    for batch in train_loader:
        mask_path = batch['mask_path']  # Extract mask paths
        del batch['mask_path']  
        del batch['human_label']

        image = batch['image'].permute(0, 3, 1, 2).to(device)  # torch.Size([4, 3, 256, 256])
        mask = batch['mask'].permute(0, 3, 1, 2).to(device)  # torch.Size([4, 1, 256, 256])

        # Create masked image
        masked_img = torch.tensor((image *(1 - mask)), dtype=torch.float32).to(device) # torch.Size([1, 3, 256, 256])

        # Encode the masked image and get the latent representation
        encoder_posterior_mask = model.encode_first_stage(masked_img)
        z_masked = model.get_first_stage_encoding(encoder_posterior_mask).detach()

        # Resize the mask to match the latent dimensions
        cc = torch.nn.functional.interpolate(mask, size=z_masked.shape[-2:])
        cond_mask = torch.cat((z_masked, cc), dim=1).squeeze(0)  # torch.Size([1, 4, 64, 64])

        # Move data to the device (GPU)
        batch["image"] = torch.tensor((127.5 * (batch['image'] + 1)), dtype=torch.uint8).to(device)
        batch["mask"] = torch.tensor(batch["mask"], dtype=torch.uint8).to(device)
        batch["feat_patch"] = batch["feat_patch"].to(device)
        batch["cond_mask"] = cond_mask.to(device)

        # Create batch for unconditional conditioning
        batch_uncond = {**batch}
        batch_uncond["feat_patch"] = torch.zeros_like(batch["feat_patch"])

        # Get conditioning
        cc = model.get_learned_conditioning(batch)
        uc = model.get_learned_conditioning(batch_uncond)

        # Sample synthetic images
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cc,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            cond_mask=batch["cond_mask"]
        )

        # Decode the samples and scale them to [0, 255]
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8).cpu()

        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).numpy()

        # Save the images
        for i in range(batch_size):
            syn_img_path = mask_path[i].replace('mask', f'image_syn_example/{index:02}')
            syn_img_path = Path(syn_img_path)
            syn_img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(x_samples_ddim[i]).save(syn_img_path)

print('finished!')