import os
import warnings
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from utils import encode_image


def plot_image_reconstruction(test_batch, reconstruction, args, print_image, model_name, num_images=5):
    """
    This function plots the reconstructed images and the original images for the test set and saves the figure in the
    results' folder.
    """
    print('Plotting the test reconstructed images')
    input_height, input_width = args.size

    # Dictionary that will store the different images and outputs for various epochs
    outputs = {'img': test_batch.to("cpu"), 'out': reconstruction.to("cpu")}

    val = [outputs['out'].permute(0, 2, 3, 1).detach().numpy(), outputs['img'].permute(0, 2, 3, 1)]

    # create 2x1 subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    # fig.suptitle('Figure title')

    # clear subplots
    for ax in axs:
        ax.remove()

    # add subfigure per subplot
    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]
    titles = ['reconstructed', 'original']
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(titles[row])

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=num_images)
        for col, ax in enumerate(axs):
            if args.color_channels == 1:
                ax.imshow(val[row][col].reshape(input_height, input_width), cmap='gray')
            else:
                ax.imshow(val[row][col])
            ax.axis('off')
            ax.set_title(f'Sample {col + 1}')

    if print_image:
        fig.show()
    fig.canvas.draw()
    # save the figure
    fig, img_name = save_figure(fig, f"{model_name}_reconstructed_test")
    X = np.array(fig.canvas.renderer.buffer_rgba())
    del outputs
    plt.close('all')
    return X


def viz_latent_space(title, model, test_dataset, args, print_image, model_name):
    """
    This function plots the latent space of the test set and saves the figure in the results' folder.
    :param title:
    :param model:
    :param test_dataset:
    :param args:
    :param print_image:
    :param model_name:
    :return:
    """
    print("Visualizing latent space with tSNE")
    data, labels = [], []
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    for sample in test_dataset:
        img = sample[0].unsqueeze(0).to(model.device)
        label = sample[1]
        # Encode image
        model.eval()
        with torch.no_grad():
            encoded_img = encode_image(img, model, inference=True)
        # Append to list
        encoded_img = encoded_img.cpu().numpy().flatten()
        data.append(encoded_img)
        labels.append(label)
    # run block of code and catch warnings
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        # execute code that will generate warnings
        print("Computing tsne representation")
        data = TSNE(init='pca', n_components=args.tsne_dim).fit_transform(np.nan_to_num(data))

    df = pd.DataFrame({"x": data[:, 0], "y": data[:, 1], "hue": labels})

    sns.scatterplot(x="x", y="y", data=df, hue="hue", legend="full", palette='colorblind').set(title=title)
    ax.legend(labels=test_dataset.classes, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # save the figure
    fig, img_name = save_figure(fig, f"{model_name}_tsne_latent_space")
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    if print_image:
        fig.show()
    del df
    plt.close('all')
    return X, img_name


# function that save the figure with specified name
def save_figure(fig, name):
    viz_path = 'visualizations'
    # Create directory if it does not exist
    if not os.path.isdir(viz_path):
        os.mkdir(viz_path)
    # crate the name of the image
    img_name = f"{name}.png"
    # check if the image already exists and if so add a number to the name
    if os.path.isfile(f"{viz_path}/{img_name}"):
        i = 1
        while os.path.isfile(f"{viz_path}/{img_name}"):
            img_name = f"{name}_{i}.png"
            i += 1
    fig.savefig(f"{viz_path}/{img_name}", bbox_inches='tight')
    plt.close('all')
    return fig, img_name
