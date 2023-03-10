import csv
import os
import sys

from model.crvae_model import CRVAE

path = os.getcwd()
sys.path.append(path)

from evaluate_utils import evaluate, denormalize
from plot_utils import plot_image_reconstruction, viz_latent_space
from utils import *
import torch
from torch import optim
from tqdm import tqdm


class Trainer:
    def __init__(self, args):
        # set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        # set the seed
        torch.manual_seed(self.args.seed)
        # set the dataset specific parameters
        self.args.color_channels = 1
        self.args.size = [28, 28]
        # get the datasets
        self.train_dataset, self.validation_dataset, self.test_dataset = get_datasets()
        # get the dataloaders
        self.train_loader, self.validation_loader, self.test_loader = get_dataloaders(self.train_dataset,
                                                                                      self.validation_dataset,
                                                                                      self.test_dataset,
                                                                                      self.args.num_workers)
        print("Training on {}".format(self.device))
        # create the model
        self.model = CRVAE(z_dim=self.args.z_dim, channels=self.args.color_channels, beta=self.args.beta,
                           gamma=self.args.gamma, K=self.args.K, m=0.99, T=0.1, device=self.device,
                           loss_type=self.args.loss).to(self.device)
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
        init_log_file(self.log_file, self.model_name)

    def train_step(self, img_pair_1):
        self.model.train()
        # get the images
        img_1, img_2 = img_pair_1
        # move the images to the device
        img_1 = img_1.cuda(non_blocking=True)
        img_2 = img_2.cuda(non_blocking=True)
        # get contrastive loss of the two views
        reconstructed, _, mu, logvar, q, k, con_loss = self.model(img_1, img_2)
        # squeeze the dimensions of the mu and logvar
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        # Calculate reconstruction loss
        reconstruction_loss = self.model.reconstruction_loss(reconstructed, img_1).sum([1, 2, 3]).mean()
        # Calculate KL divergence
        kld = kl_divergence(mu, logvar)
        # Calculate the contrastive loss InfoNCE
        contrastive_loss = con_loss.item()
        # compute the total loss
        loss = self.alpha * reconstruction_loss + self.beta * kld + self.gamma * con_loss
        # The gradients are set to zero
        self.optimizer.zero_grad()
        # compute the gradients w.r.t. elbo loss
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        # update the parameters
        self.optimizer.step()
        return loss.item(), reconstruction_loss.item(), kld.item(), contrastive_loss

    def train_loop(self):
        # resume from a model checkpoint
        if self.args.resume:
            checkpoint = torch.load(self.args.load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 1

        for epoch in range(start_epoch, self.args.epochs + 1):
            # set model parameters to trainable
            self.model.train()
            # initialize the training bar
            train_bar = tqdm(self.train_loader, position=0, leave=True)
            # iterate over the training data
            for img_pair in train_bar:
                # perform a training step
                self.loss, bce_loss, kld, self.contrastive_loss = self.train_step(img_pair)
                # update the training losses
                self.total_loss += self.loss
                self.total_reconstruction_loss += bce_loss
                self.total_kl_loss += kld
                self.total_contrastive_loss += self.contrastive_loss
                # update the progress bar
                train_bar.set_description(
                    'Epoch: [{}/{}] '
                    'lr: {:.6f} '
                    'Loss: {:.4f} '
                    'Rec/tion: {:.4f} '
                    'KL: {:.4f} '
                    'NCE: {:.4f} '
                    'alpha: {:.4f} '
                    'beta: {:.4f} '
                    'gamma: {:.4f} '
                    'delta: {:.4f} '.format(epoch, self.args.epochs, self.args.learning_rate, self.loss, bce_loss, kld,
                                            self.contrastive_loss, self.args.alpha, self.args.beta,
                                            self.args.gamma, self.args.delta))
            # evaluate the model
            self.evaluate(epoch)

    def evaluate(self, epoch):
        # Averaging out loss over entire batch
        num_of_batches = len(self.validation_loader)
        self.total_loss /= num_of_batches
        self.total_reconstruction_loss /= num_of_batches
        self.total_kl_loss /= num_of_batches
        self.total_contrastive_loss /= num_of_batches

        # update lr
        self.scheduler.step(self.total_loss)
        lr = get_lr(self.optimizer)

        # store losses to lists
        self.train_loss_list.append(self.total_loss)
        self.train_reconstruction_loss_list.append(self.total_reconstruction_loss)
        self.train_kl_list.append(self.total_kl_loss)
        self.train_contrastive_list.append(self.contrastive_loss)

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
            plot_image_reconstruction(test_batch, reconstruction, self.args, print_image=True,
                                      model_name=self.model_name)
        # final epoch
        if epoch == self.args.epochs:
            # plot the latent space of the test data using t-SNE
            tsne_title = f"Test data latent space at epoch {epoch}"
            viz_latent_space(tsne_title, self.model, self.test_dataset, self.args,
                             print_image=True, model_name=self.model_name)
