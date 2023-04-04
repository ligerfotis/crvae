import csv
import os

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import EMNIST, MNIST
from tqdm import tqdm

# definitions of train and test transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((28, 28)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.ToTensor()])


def get_args(parser):
    """
    This function adds the arguments to the parser
    :param parser: # the parser to add the arguments to
    :return: # the parser with the added arguments
    """
    # the dataset to train on
    parser.add_argument('-d', '--dataset', default="EMNIST", type=str)
    # the size of the latent space representation
    parser.add_argument('-zd', '--z-dim', default=64, type=int,
                        help='the dimension of the latent space representation')
    # input image channel size
    parser.add_argument('-c', '--channel-size', default=3, type=int,
                        help='the channel size of the input')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print things')
    parser.add_argument('-r', '--resume', action='store_true', help='resume training from path given')
    # argument to use wandb
    parser.add_argument('-w', '--wandb', action='store_true', help='use wandb')
    # argument for model size
    parser.add_argument('-ms', '--model-size', type=str, default='small', help='size of the model')
    # argument for augmnentation
    parser.add_argument('-aug', '--augment', action='store_true', help='use data augmentation')

    parser.add_argument('-bs', '--batch-size', default=512, type=int)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-lr', '--learning-rate', default=1e-5, type=float)
    parser.add_argument('-nw', '--num-workers', default=16, type=int)
    parser.add_argument('-si', '--save_integral', default=10, type=int)
    # parser.add_argument('-lp', '--load-path', type=str)
    # parser.add_argument('-ckp', '--checkpoint', action='store_true',
    #                     help='load model checkpoint. If false model dictionary is loaded')
    # parser.add_argument('-n', '--model-name', type=str)
    # parser.add_argument('-rp', '--run-path', type=str)
    # loss function to use
    parser.add_argument('-l', '--loss', default='bce',
                        help=' reconstruction loss function to use. currently supporting mse and bce')
    parser.add_argument('-ni', '--norm-input', action='store_true',
                        help='normalize input')
    parser.add_argument('-sh', '--shuffle', action='store_true',
                        help='shuffle train dataloader')
    parser.add_argument('-opt', '--optimizer', default='sgd',
                        help='which optimizer to use')
    parser.add_argument('-wd', '--weight-decay', default=1e-8, type=float, help='weight decay for optimizer')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='momentum value for sgd optimizer')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to moco pretrained checkpoint')
    parser.add_argument('-aut', '--active-units-threshold', default=0.01, type=float, help='active units threshold')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tsne_dim', default=2, type=int)
    parser.add_argument('--knn_k', default=200, type=int, help='number of voters in the KNN algorithm')
    parser.add_argument('--knn_t', default=0.1, type=float, help='temperature parameter for the weighting in KNN')
    parser.add_argument('--alpha', default=1, type=float, help='weight of reconstruction term')
    parser.add_argument('--beta', default=1.0, type=float, help='weight of KL term')
    parser.add_argument('--gamma', default=1, type=float, help='weight of infonce term')
    parser.add_argument('--delta', default=0, type=float, help='InfoMax temperature')
    parser.add_argument('--d_lr', default=1e-3, type=float, help='learning rate for discriminator')
    parser.add_argument('--K', default=4096, type=int, help='number of negative samples')
    parser.add_argument('--representation_metrics', default=1, type=int, help='produce representation metrics')
    args = parser.parse_args()  # running in command line

    return args

class EMNISTPair(EMNIST):
    """
    This is a modified version of the EMNIST dataset class from torchvision.
    It returns a pair of stochastic augmentations of an image.
    """

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img.to("cpu").detach().numpy())

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2

class MNISTPair(MNIST):
    """
    This is a modified version of the MNIST dataset class from torchvision.
    It returns a pair of stochastic augmentations of an image.
    """

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img.to("cpu").detach().numpy())

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


def get_optimizer(model_parameteres, args):
    """
    This function returns the optimizer based on the arguments.
    """
    if args.optimizer in ["adam", "Adam", "ADAM"]:
        return optim.Adam(model_parameteres, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer in ["sgd", "SGD", "Sdg"]:
        return optim.SGD(model_parameteres, lr=args.learning_rate, weight_decay=args.weight_decay,
                         momentum=args.momentum)
    else:
        raise Exception("unknown optimizer asked in \"get_optimizer()\"")


def kl_divergence(mu, logvar):
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()


def get_lr(optimizer):
    """Get learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Discriminator(nn.Module):
    """
    Discriminator network for computing the mutual information between the latent space and the data for InfoMax VAE.
    """

    def __init__(self, args=None):
        super(Discriminator, self).__init__()
        self.channels = args.color_channels
        self.height, self.width = args.size
        self.z_dim = args.z_dim
        self.net = nn.Sequential(
            nn.Linear(self.channels * self.height * self.width + self.z_dim, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

    def forward(self, x, z):
        x = x.view(-1, self.channels * self.height * self.width)
        x = torch.cat((x, z), 1)
        pred = self.net(x).squeeze() if self.channels > 1 else self.net(x)
        return pred


def get_loss(model, reconstructed, img, mu, logvar):
    """
    This function computes the loss for the VAE.
    """
    reconstruction_loss = model.reconstruction_loss(reconstructed, img).sum([1, 2, 3]).mean()
    kld = kl_divergence(mu, logvar)
    vae_loss = reconstruction_loss + model.beta * kld

    return vae_loss, reconstruction_loss, kld


def calc_mi(model, test_loader, device):
    """
    adjusted from (https://github.com/jxhe/vae-lagging-encoder)
    compute the mutual information between the latent space and the data
    """
    mi = 0
    num_examples = 0
    for datum in test_loader:
        batch_data, _ = datum
        batch_data = batch_data.to(device)
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.encoder.calc_mi(batch_data)
        mi += mutual_info * batch_size

    return - mi / num_examples


def calc_au(model, test_data_batch, delta=0.01):
    """
    adjusted from (https://github.com/jxhe/vae-lagging-encoder)
    compute the number of active units
    """
    means_sum, var_sum = None, None
    cnt = 0
    for batch_data in test_data_batch:
        if isinstance(batch_data, list):
            batch_data = batch_data[0]
        _, mu, _, _ = model.predict_batch(batch_data)
        if means_sum is None:
            means_sum = mu.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mu.sum(dim=0, keepdim=True)
        cnt += mu.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch_data in test_data_batch:
        if isinstance(batch_data, list):
            batch_data = batch_data[0]
        _, mu, _, _ = model.predict_batch(batch_data)
        if var_sum is None:
            var_sum = ((mu - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mu - mean_mean) ** 2).sum(dim=0)
        cnt += mu.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    return (au_var >= delta).sum().item(), au_var


def save_model(checkpoint, model_name, args):
    paths = ["model/model_checkpoints/"]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    save_ckp(checkpoint, f"model/model_checkpoints/{model_name}")


def save_ckp(state, checkpoint_dir):
    f_path = f'{checkpoint_dir}.ckp'
    torch.save(state, f_path)


def representation_metric_test(net, memory_data_loader, test_data_loader, knn_k, epoch, inference=True):
    """
    This function is used to test the representation metric of the model. We use the trained model in two
    semi-supervised learning tasks: 1) classification on the test set; 2) kNN classification on the memory set. The
    performance of the model in these two tasks is used to evaluate the representation metric of the model.
    """
    net.eval()
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        feature_bar = tqdm(memory_data_loader, position=0, leave=True, desc='Feature extracting')
        for data, target in feature_bar:
            feature = encode_image(data, net, inference=inference)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if isinstance(memory_data_loader.dataset.targets, list):
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        elif isinstance(memory_data_loader.dataset.targets, torch.Tensor):
            feature_labels = memory_data_loader.dataset.targets.clone().detach().to(feature_bank.device)

        # convert to numpy
        feature_bank_np = feature_bank.t().cpu().detach().numpy()
        feature_labels_np = feature_labels.cpu().detach().numpy()[:len(feature_bank_np)]

        # create linear classifier
        classifier = SGDClassifier(loss='perceptron', n_jobs=-1)
        print('\nFitting linear classifier')
        # fit linear classifier
        classifier.fit(feature_bank_np, feature_labels_np)
        # create knn classifier
        neigh = KNeighborsClassifier(n_neighbors=knn_k, n_jobs=-1)
        print('\nFitting KNN classifier')
        # fit knn classifier
        neigh.fit(feature_bank_np, feature_labels_np)
        # loop test data to predict the label by weighted knn search
        feature_bank_np_test, feature_labels_np_test = [], []
        test_bar = tqdm(test_data_loader, position=0, leave=True)
        for data, target in test_bar:
            # send to device
            target = target.cuda(non_blocking=True)
            # get feature
            feature = encode_image(data, net, inference=inference)
            # append to list
            feature_bank_np_test.append(feature.cpu().detach().numpy())
            feature_labels_np_test.append(target.cpu().detach().numpy())
            # prediction using knn
            pred_labels = neigh.predict(feature.cpu().detach().numpy())

            total_num += data.size(0)
            total_top1 += (pred_labels == target.cpu().numpy()).sum().item()
            knn = total_top1 / total_num * 100

            test_bar.set_description("KNN classification test Epoch {}: Acc@1:{:.2f}%".format(epoch, knn),
                                     refresh=True)
        feature_labels_np_test = np.concatenate(feature_labels_np_test, axis=0)
        feature_bank_np_test = np.concatenate(feature_bank_np_test, axis=0)
    y_pred = classifier.predict(feature_bank_np_test)
    result_linear = np.mean(y_pred == feature_labels_np_test) * 100

    return knn, result_linear


def encode_image(batch, model, inference=False):
    """
    This function is used to encode the image into the latent space.
    """
    data = batch.cuda(non_blocking=True)
    feature, feature_mu, feature_logvar = model.encoder(data)
    if inference:
        feature = feature_mu
    return feature


def recon_loss(loss):
    """
    This function is used to get the reconstruction loss function.
    """
    if loss in ["BCE", 'bce']:
        loss = nn.BCELoss(reduction='none')
    elif loss in ["MSE", 'mse']:
        loss = nn.MSELoss(reduction='none')
    return loss


def get_dataloaders(train_dataset, memory_data, test_dataset, num_workers=4):
    """
    # get the data loaders for the training, validation and test datasets
    :param num_workers: number of workers for the data loader
    :return: train_loader, validation_loader, test_loader
    """
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    validation_loader = torch.utils.data.DataLoader(memory_data, batch_size=512, shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)
    return train_loader, validation_loader, test_loader


def get_datasets(dataset, augment=False):
    """
    # get the train, validation and test datasets
    :return: train_dataset, memory_data, test_dataset
    """
    if dataset == 'EMNIST':
        # root folder for the dataset
        root = "./data/EMNIST"
        # get the train dataset
        if augment:
            train_dataset = EMNISTPair(root=f"{root}/train", train=True, transform=train_transform, split='balanced',
                                  download=True)
        else:
            train_dataset = EMNIST(root=f"{root}/train", train=True, transform=test_transform, split='balanced',
                                  download=True)
        # get the train dataset with the test transform for validation in the semisupevised setting
        memory_data = EMNIST(root=f"{root}/train", train=True, transform=test_transform, split='balanced',
                            download=True)
        # get the test dataset
        test_dataset = datasets.EMNIST(root=f"{root}/test", train=False, transform=test_transform, split='balanced',
                                      download=True)
    elif dataset == 'MNIST':
        # root folder for the dataset
        root = "./data/MNIST"
        # get the train dataset
        if augment:
            train_dataset = MNISTPair(root=f"{root}/train", train=True, transform=train_transform, download=True)
        else:
            train_dataset = MNISTPair(root=f"{root}/train", train=True, transform=test_transform, download=True)
        # get the train dataset with the test transform for validation in the semisupevised setting
        memory_data = datasets.MNIST(root=f"{root}/train", train=True, transform=test_transform, download=True)
        # get the test dataset
        test_dataset = datasets.MNIST(root=f"{root}/test", train=False, transform=test_transform, download=True)
    return train_dataset, memory_data, test_dataset


def init_log_file(log_file, model_name):
    """
    This function is used to initialize the log file.
    """
    # check if the log file exists
    if not os.path.exists("logs"):
        os.mkdir("logs")
    # check if file with filename exists
    if not os.path.exists(log_file):
        # create the log file and write the header
        with open(log_file, 'w') as f:
            log_writer = csv.writer(f)
            log_writer.writerow(
                ['epoch', 'lr', 'train_loss', 'train_reconstruction_loss', 'train_kl_loss',
                 'train_contrastive_loss', 'KNN_acc',
                 'linear_acc', 'alpha', 'beta', 'gamma', 'delta', 'mi', 'au', 'eval_loss', 'eval_recon', 'eval_kl'])
    else:
        # change the name of the file to avoid overwriting by adding a number
        i = 1
        while os.path.exists(log_file):
            log_file = f"logs/{model_name}_logs_{i}.csv"
            i += 1
        # replace the file with a new one
        with open(log_file, 'w') as f:
            log_writer = csv.writer(f)
            log_writer.writerow(
                ['epoch', 'lr', 'train_loss', 'train_reconstruction_loss', 'train_kl_loss',
                 'train_contrastive_loss', 'KNN_acc',
                 'linear_acc', 'alpha', 'beta', 'gamma', 'delta', 'mi', 'au', 'eval_loss', 'eval_recon', 'eval_kl'])
    return log_file
