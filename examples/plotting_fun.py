import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pytorch_laplace.laplace.diag import DiagLaplace


def plot_digit(img):
    plt.figure()
    plt.imshow(img[0].detach().numpy())


def plot_latent(encoder, dataset, posterior=(0,1), scale_radius=1, n_sample=1000):
    mean, std_deviation = posterior
    plt.figure(figsize=(10,10))

    # plot MAP parameter latent
    vector_to_parameters(mean, encoder.parameters())
    latent = encoder(dataset).detach().numpy()
    plt.scatter(latent[:,0], latent[:,1], label='MAP prediction')

    # plot statistics (mean and std) of latent of samples following the posterior
    latent_list = []
    sampler = DiagLaplace()
    samples = sampler.sample(mean, std_deviation, n_samples=n_sample)
    for parameter in samples:
        vector_to_parameters(parameter, encoder.parameters())
        latent = encoder(dataset)
        latent_list.append(latent)
    latent_list = torch.stack(latent_list,dim=0)
    latent_mean = torch.mean(latent_list, dim=0).detach().numpy()
    latent_std = torch.std(latent_list, dim=0).detach().numpy()
    plt.scatter(latent_mean[:,0], latent_mean[:,1], label='prediction samples')
    for b in range(len(dataset)):
        circle = Ellipse((latent_mean[b, 0], latent_mean[b, 1]), 
                        width = scale_radius * latent_std[b, 0], 
                        height = scale_radius * latent_std[b, 1], 
                        color='r', alpha=0.1)
        plt.gca().add_patch(circle)
    plt.legend()
    plt.xlim([-1.05,1.05])
    plt.ylim([-1.05,1.05])
    plt.title('Latent space')


def plot_fancy_latent(encoder, batch):
    latent = encoder(batch).detach().numpy()
    zoom = 2
    fig, ax = plt.subplots(figsize=(10,10))
    for i,image in enumerate(batch):
        im = OffsetImage(image[0], zoom=zoom)
        ab = AnnotationBbox(im, (latent[i][0], latent[i][1]), xycoords='data', frameon=False)
        ax.add_artist(ab)
        ax.update_datalim([(latent[i][0], latent[i][1])])
        ax.autoscale()
    plt.title('Latent space')


def plot_reconstruction(model, dataset, posterior=(0,1), data_idx=0, n_sample=1000, apply_softmax=False):
    mean, std_deviation = posterior
    plt.figure(figsize=(15,3))

    # plot target label
    ax = plt.subplot(1, 4, 1)
    plt.imshow(dataset[data_idx][0].detach().numpy())
    ax.title.set_text('Ground truth')

    # compute MAP parameter predictions
    vector_to_parameters(mean, model.parameters())
    if apply_softmax:
        logits = model(dataset)
        exp_logits = torch.exp(logits)
        predictions = exp_logits / (1 + exp_logits)
    else:
        predictions = model(dataset)
    ax = plt.subplot(1, 4, 2)
    plt.imshow(predictions[data_idx][0].detach().numpy())
    ax.title.set_text('MAP prediction')

    # plot statistics (mean and std) of prediction of samples following the posterior
    predictions_list = []
    sampler = DiagLaplace()
    samples = sampler.sample(mean, std_deviation, n_samples=n_sample)
    for parameter in samples:
        vector_to_parameters(parameter, model.parameters())
        if apply_softmax:
            logits = model(dataset)
            exp_logits = torch.exp(logits)
            softmax_probs = exp_logits / (1 + exp_logits)
            predictions_list.append(softmax_probs)
        else:
            predictions = model(dataset)
            predictions_list.append(predictions)
    predictions_list = torch.stack(predictions_list,dim=0)
    ax = plt.subplot(1, 4, 3)
    plt.imshow(torch.mean(predictions_list, dim=0)[data_idx][0].detach().numpy())
    ax.title.set_text('Predictions mean')
    ax = plt.subplot(1, 4, 4)
    plt.imshow(torch.std(predictions_list, dim=0)[data_idx][0].detach().numpy())
    ax.title.set_text('Predictions std')


def plot_reconstruction_with_latent(model, dataset, posterior=(0,1), data_idx=0, scale_radius=1, n_sample=1000, apply_softmax=False):
    mean, std_deviation = posterior
    plt.figure(figsize=(15,3))

    # plot target label
    ax = plt.subplot(1, 5, 1)
    plt.imshow(dataset[data_idx][0].detach().numpy())
    ax.title.set_text('Ground truth')

    # plot latent
    ax = plt.subplot(1, 5, 2)
    encoder = model[0]
        # plot MAP parameter latent
    vector_to_parameters(mean, model.parameters())
    latent = encoder(dataset).detach().numpy()
    plt.scatter(latent[data_idx,0], latent[data_idx,1], label='MAP prediction')
        # plot statistics (mean and std) of latent of samples following the posterior
    latent_list = []
    sampler = DiagLaplace()
    samples = sampler.sample(mean, std_deviation, n_samples=n_sample)
    for parameter in samples:
        vector_to_parameters(parameter, model.parameters())
        latent = encoder(dataset)
        latent_list.append(latent)
    latent_list = torch.stack(latent_list,dim=0)
    latent_mean = torch.mean(latent_list, dim=0).detach().numpy()
    latent_std = torch.std(latent_list, dim=0).detach().numpy()
    plt.scatter(latent_mean[data_idx,0], latent_mean[data_idx,1], label='prediction samples')
    circle = Ellipse((latent_mean[data_idx, 0], latent_mean[data_idx, 1]), 
                        width = scale_radius * latent_std[data_idx, 0], 
                        height = scale_radius * latent_std[data_idx, 1], 
                        color='r', alpha=0.1)
    plt.gca().add_patch(circle)
    #plt.legend()
    plt.xlim([-1.05,1.05])
    plt.ylim([-1.05,1.05])
    ax.title.set_text('Latent')

    # compute MAP parameter predictions
    vector_to_parameters(mean, model.parameters())
    if apply_softmax:
        logits = model(dataset)
        exp_logits = torch.exp(logits)
        predictions = exp_logits / (1 + exp_logits)
    else:
        predictions = model(dataset)
    ax = plt.subplot(1, 5, 3)
    plt.imshow(predictions[data_idx][0].detach().numpy())
    ax.title.set_text('MAP prediction')

    # plot statistics (mean and std) of prediction of samples following the posterior
    predictions_list = []
    sampler = DiagLaplace()
    samples = sampler.sample(mean, std_deviation, n_samples=n_sample)
    for parameter in samples:
        vector_to_parameters(parameter, model.parameters())
        if apply_softmax:
            logits = model(dataset)
            exp_logits = torch.exp(logits)
            softmax_probs = exp_logits / (1 + exp_logits)
            predictions_list.append(softmax_probs)
        else:
            predictions = model(dataset)
            predictions_list.append(predictions)
    predictions_list = torch.stack(predictions_list,dim=0)
    ax = plt.subplot(1, 5, 4)
    plt.imshow(torch.mean(predictions_list, dim=0)[data_idx][0].detach().numpy())
    ax.title.set_text('Predictions mean')
    ax = plt.subplot(1, 5, 5)
    plt.imshow(torch.std(predictions_list, dim=0)[data_idx][0].detach().numpy())
    ax.title.set_text('Predictions std')


def plot_std(std_deviation, encoder_size, decoder_size):

    plt.figure(figsize=(30,6))
    plt.plot(std_deviation[:encoder_size], label='Encoder', alpha=1)
    plt.plot(np.arange(encoder_size, encoder_size+decoder_size), 
            std_deviation[-decoder_size:], label='Decoder', alpha=1)
    plt.yscale('log')
    plt.legend()
    plt.title('Standard deviation over parameters')

    plt.figure(figsize=(30,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange( encoder_size - 102, encoder_size), 
            std_deviation[encoder_size - 102 : encoder_size], label='Encoder', alpha=1, c='C0')
    plt.yscale('log')
    plt.title('Second layer of Encoder')
    plt.subplot(1,2,2)
    plt.plot(np.arange(encoder_size, encoder_size + 150), 
            std_deviation[encoder_size : encoder_size + 150], label='Decoder', alpha=1, c='C1')
    plt.yscale('log')
    plt.title('First layer of Decoder')

    plt.figure(figsize=(30,6))
    plt.subplot(1,2,1)
    params = std_deviation[: 28*28 * 50].reshape(50, 28*28).transpose(0,1).reshape(28*28 * 50)
    plt.plot(params, label='Encoder', alpha=1, c='C0')
    plt.yscale('log')
    plt.title('First layer of Encoder (per pixel)')
    plt.subplot(1,2,2)
    params = std_deviation[-28*28 * (50+1) : -28*28]
    plt.plot(params, label='Decoder', alpha=1, c='C1')
    plt.yscale('log')
    plt.title('Second layer of Decoder')


def plot_attention(std_deviation, log_scale=False):
    plt.figure(figsize=(10,10))

    params = std_deviation[: 28*28 * 50].reshape(50, 28*28)
    if log_scale:
        params = torch.log(params)
    attention_avg = torch.mean(params, dim=0).reshape(28,28)
    attention_std = torch.std(params, dim=0).reshape(28,28)
    plt.subplot(2,2,1)
    plt.imshow(attention_avg.detach().numpy())
    plt.title('First layer of Encoder (per pixel) mean')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(attention_std.detach().numpy())
    plt.title('First layer of Encoder (per pixel) std')
    plt.colorbar()

    params = std_deviation[-28*28 * (50+1) : -28*28].reshape(28 * 28, 50)
    if log_scale:
        params = torch.log(params)
    attention_avg = torch.mean(params, dim=1).reshape(28,28)
    attention_std = torch.std(params, dim=1).reshape(28,28)
    plt.subplot(2,2,2)
    plt.imshow(attention_avg.detach().numpy())
    plt.title('Second layer of Encoder (per pixel) mean')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(attention_std.detach().numpy())
    plt.title('Second layer of Encoder (per pixel) std')
    plt.colorbar()


def plot_training(losses, losses_test, priors):
    f = plt.figure(figsize=(9,7))
    plt.suptitle('Loss and prior during training')
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5,1])
    ax1 = plt.subplot(gs[0])
    ax1.plot(losses, label='Train Loss')
    ax1.plot(losses_test, label='Test Loss')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2 = plt.subplot(gs[1])
    ax2.plot(priors, label='log prior')
    ax2.set_yscale('log')
    ax2.set_xscale('log')