import csv
import os
import sys

from model.crvae_model import CRVAE
import wandb

from model.vae_model import VAE

path = os.getcwd()
sys.path.append(path)

from evaluate_utils import evaluate, denormalize
from plot_utils import plot_image_reconstruction, viz_latent_space
from utils import *
import torch
from torch import optim
from tqdm import tqdm


class Infomax_VAE_Trainer:
    def __init__(self, args):
        # set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        # set the seed
        torch.manual_seed(self.args.seed)
        # set the dataset specific parameters
        self.args.color_channels = 1 if self.args.dataset in ['MNIST', 'FashionMNIST', 'EMNIST'] else 3
        self.args.size = [28, 28] if self.args.dataset in ['MNIST', 'FashionMNIST', 'EMNIST'] else [32, 32]
        # get the datasets
        self.train_dataset, self.validation_dataset, self.test_dataset = get_datasets(self.args.dataset,
                                                                                      self.args.augment)
        # get the dataloaders
        self.train_loader, self.validation_loader, self.test_loader = get_dataloaders(self.train_dataset,
                                                                                      self.validation_dataset,
                                                                                      self.test_dataset,
                                                                                      self.args.num_workers)
        print("Training on {}".format(self.device))
        # create the model
        self.model = VAE(z_dim=self.args.z_dim, channels=self.args.color_channels, beta=self.args.beta,
                         device=self.device, loss_type=self.args.loss, model_size=self.args.model_size,

                         output_size=self.args.size[0]).to(self.device)
        if self.args.verbose:
            print(self.model)
        # initialize the optimizer
        self.optimizer = get_optimizer(self.model.parameters(), self.args)
        # get the total number of trainable parameters
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model's total number of trainable parameters:{pytorch_total_params}\n")
        # retrieve the name of the model
        self.model_name = f"{self.args.dataset}_beta{self.args.beta}_gamma{self.args.gamma}_crvae_custom_l{self.args.z_dim}_s{self.args.seed}"
        # Create a discriminator model. It wil be used to compute the mutual information (MI) between an image and
        # its representation
        self.D = Discriminator(self.args).to(self.model.device)
        # initialize the Disctiminator optimizers
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.args.d_lr, betas=(0.9, 0.999))
        # Lists that will store the training losses
        self.train_loss_list, self.train_reconstruction_loss_list, self.train_kl_list, self.train_contrastive_list = [], [], [], []
        # scheduler for the learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20,
                                                              min_lr=5e-6)
        # Initializing variable for storing loss
        self.total_loss, self.total_reconstruction_loss, self.total_kl_loss, self.total_contrastive_loss = 0, 0, 0, 0
        # set the weights of the loss terms
        self.alpha, self.beta, self.gamma = self.args.alpha, self.args.beta, self.args.gamma
        # file to store the logs
        self.log_file = f"logs/{self.model_name}_logs.csv"
        # initialize the log file
        self.log_file = init_log_file(self.log_file, self.model_name)

    def train_step(self, img_org):
        self.model.train()
        # move the images to the device
        img_org = img_org.cuda(non_blocking=True)
        # get contrastive loss of the two views
        reconstructed, z, mu, logvar = self.model(img_org)
        # squeeze the dimensions of the mu and logvar
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        # Calculate reconstruction loss
        reconstruction_loss = self.model.reconstruction_loss(reconstructed, img_org).sum([1, 2, 3]).mean()
        # Calculate KL divergence
        kld = kl_divergence(mu, logvar)
        # detach the latent representation
        z_detach = z.detach()
        # compute mutial information I(x, z)
        MI_x_z = compute_mutual_information(img_org, z_detach, self.D, self.optim_D, self.model.device)
        # compute the total loss
        loss = self.alpha * reconstruction_loss + self.beta * kld + self.gamma * MI_x_z
        # The gradients are set to zero
        self.optimizer.zero_grad()
        # compute the gradients w.r.t. elbo loss
        loss.backward()
        # update the parameters
        self.optimizer.step()
        return loss.item(), reconstruction_loss.item(), kld.item(), MI_x_z

    def train_loop(self):
        # resume from a model checkpoint
        if self.args.resume:
            checkpoint = torch.load(self.args.load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 1
        # initialize wandb logging tool
        if self.args.wandb:
            wandb.init(project="cr-vae", config=self.args)
            # change default run name
            wandb.run.name = self.model_name
            # save training configuration
            wandb.config.update(self.args)
        # iterate over the epochs
        for epoch in range(start_epoch, self.args.epochs + 1):
            # set model parameters to trainable
            self.model.train()
            # initialize the training bar
            train_bar = tqdm(zip(self.train_loader, self.validation_loader), position=0, leave=True)
            # iterate over the training data
            for img_pairs in train_bar:
                if self.args.augment:
                    img_org = img_pairs[0][0]
                else:
                    img_org = img_pairs[1][0]
                # perform a training step
                self.loss, bce_loss, kld, mi_d = self.train_step(img_org)
                # update the training losses
                self.total_loss += self.loss
                self.total_reconstruction_loss += bce_loss
                self.total_kl_loss += kld
                # update the progress bar
                train_bar.set_description(
                    'Epoch: [{}/{}] '
                    'lr: {:.6f} '
                    'Loss: {:.4f} '
                    'Rec/tion: {:.4f} '
                    'KL: {:.4f} '
                    'alpha: {:.4f} '
                    'beta: {:.4f} '
                    'gamma: {:.4f} '
                    'mi_d: {:.4f} '.format(epoch, self.args.epochs, self.args.learning_rate, self.loss, bce_loss, kld,
                                            self.args.alpha, self.args.beta,
                                            self.args.gamma, mi_d))
            # evaluate the model
            self.evaluate(epoch, wandb_log=self.args.wandb)
        wandb.finish()

    def evaluate(self, epoch, wandb_log=True):
        # Averaging out loss over entire batch
        num_of_batches = len(self.validation_loader)
        self.total_loss /= num_of_batches
        self.total_reconstruction_loss /= num_of_batches
        self.total_kl_loss /= num_of_batches

        # update lr
        self.scheduler.step(self.total_loss)
        lr = get_lr(self.optimizer)

        # store losses to lists
        self.train_loss_list.append(self.total_loss)
        self.train_reconstruction_loss_list.append(self.total_reconstruction_loss)
        self.train_kl_list.append(self.total_kl_loss)

        # store the model parameters to a dictionary for saving the model checkpoint
        checkpoint = {
            'epoch': epoch,
            'latent_space_dimension': self.args.z_dim,
            'channels': self.args.color_channels,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'args': self.args
        }
        self.model.eval()
        if epoch % self.args.save_integral == 0:
            # evaluate the model on the test set
            eval_loss, eval_mse, eval_kl = evaluate(self.test_loader, self.model, self.args)
            with torch.no_grad():
                # calculate mutual information between latent space and input
                mi = calc_mi(self.model, self.test_loader, self.device)
                # calculate the activation units
                au, _ = calc_au(self.model, self.test_loader)
            if self.args.verbose:
                print("Saving model. Epoch {}".format(epoch))
            save_model(checkpoint, self.model_name, self.args)
            print(f"Test: Rec/tion:{eval_mse:.2f} | KL:{eval_kl:.2f} | MI:{mi:.2f} | AU:{au:.2f}")

            # get the last batch of the test loader
            test_batch, _ = list(self.test_loader)[-1]
            # move the batch to the device
            with torch.no_grad():
                # reconstruct the images in the batch
                reconstructed, _, _, z = self.model.predict_batch(test_batch)
            # denormalize the images
            test_batch = denormalize(test_batch) if self.args.norm_input else test_batch
            reconstruction = denormalize(reconstructed) if self.args.norm_input else reconstructed
            # evaluate the representations of the model on semi-supervised tasks
            if self.args.representation_metrics == 1:
                knn_acc, linear_acc = representation_metric_test(self.model,
                                                                 self.validation_loader,
                                                                 self.test_loader, self.args.knn_k, epoch,
                                                                 inference=True)
            else:
                # in case we don't want to evaluate the representations (less computation time
                knn_acc, linear_acc = 0, 0

            # print the accuracy
            print(f"Linear Classification: {linear_acc:.2f}%")
            print(f"KNN Classification: {knn_acc:.2f}%")

            # Save logs to a csv file that has name for columns
            with open(self.log_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, lr, self.total_loss, self.total_reconstruction_loss, self.total_kl_loss,
                                 self.total_contrastive_loss, knn_acc, linear_acc, self.args.alpha, self.args.beta,
                                 self.args.gamma, self.args.delta, mi, au, eval_loss, eval_mse.item(),
                                 eval_kl.item()])
            # plot the image reconstruction
            reconstructed_img_array = plot_image_reconstruction(test_batch, reconstruction, self.args, print_image=True,
                                                                model_name=self.model_name)
            reconstructed_name = f"Test reconstruction epoch at {epoch} with loss {eval_loss}"
            reconstructed_image = wandb.Image(reconstructed_img_array, caption=reconstructed_name)
            # Save logs in wandb
            if wandb_log:
                wandb.log({"epoch": epoch,
                           "lr": lr,
                           "train_loss": self.total_loss,
                           "train_reconstruction_loss": self.total_reconstruction_loss,
                           "train_kl_loss": self.total_kl_loss,
                           "train_contrastive_loss": self.total_contrastive_loss,
                           "knn_acc": knn_acc,
                           "linear_acc": linear_acc,
                           "alpha": self.args.alpha,
                           "beta": self.args.beta,
                           "gamma": self.args.gamma,
                           "delta": self.args.delta,
                           "mi": mi,
                           "au": au,
                           "test_loss": eval_loss,
                           "test_reconstruction_loss": eval_mse.item(),
                           "test_kl_loss": eval_kl.item(),
                           "reconstructed_img": reconstructed_image})

        # final epoch
        if epoch == self.args.epochs:
            # plot the latent space of the test data using t-SNE
            tsne_title = f"Test data latent space at epoch {epoch}"
            tsne_img_array, tsne_name = viz_latent_space(tsne_title, self.model, self.test_dataset, self.args,
                                                         print_image=True, model_name=self.model_name)
            if wandb_log:
                # Save latent space in wandb
                wandb.log({"latent_space": wandb.Image(tsne_img_array, tsne_name)})


def compute_mutual_information(img_1, z_detach, D, optim_D, device):
    D_xz = D(img_1, z_detach)
    z_perm = permute_dims(z_detach, device)
    D_x_z = D(img_1, z_perm)

    Info_xz = -(D_xz.mean() - (torch.exp(D_x_z - 1).mean()))

    info_loss = Info_xz
    if optim_D is not None:
        optim_D.zero_grad()
        info_loss.backward()
        optim_D.step()

    return info_loss.item()


def permute_dims(z, device):
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B).to(device)
    perm_z = z[perm]
    return perm_z
