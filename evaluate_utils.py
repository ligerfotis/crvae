import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from utils import get_loss


def denormalize(normalized_tensor):
    # denormalize the tensor
    invTrans = transforms.Compose([transforms.Normalize(mean=[-0.485 / 0.229],
                                                        std=[1 / 0.229])])
    return invTrans(normalized_tensor)


def evaluate(test_loader, model, args):
    total_loss, total_reconstruction_loss, total_kl_loss = 0, 0, 0

    for batch in tqdm(test_loader, position=0, leave=True):
        img_1, _ = batch
        # reconstruct an image batch
        with torch.no_grad():
            reconstructed, mu, var, z = model.predict_batch(img_1)
            img_1_denorm = denormalize(img_1) if args.norm_input else img_1
            reconstructed = denormalize(reconstructed) if args.norm_input else reconstructed
            # send the batch of images to the device
            img_1_denorm = img_1_denorm.to(model.device)
            # Calculating the loss function
            eval_loss, mse_loss, kl_loss = get_loss(model, reconstructed, img_1_denorm, mu, var)

        # Incrementing losses
        total_loss += eval_loss.item()
        total_reconstruction_loss += mse_loss
        total_kl_loss += kl_loss

    # number of batches
    num_batches = len(test_loader)
    # Averaging out loss over entire batch
    total_loss /= num_batches
    total_reconstruction_loss /= num_batches
    total_kl_loss /= num_batches

    print(f"Evaluate: Total loss:{total_loss:.3f}|MSE:{total_reconstruction_loss:.3f}|KL:{total_kl_loss:.3f}")

    return total_loss, total_reconstruction_loss, total_kl_loss
