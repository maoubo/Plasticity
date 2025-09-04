import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.autograd import grad
from matplotlib.colors import LinearSegmentedColormap

def plasticity_judge(update, num_updates, epoch, start):
    if epoch == 0 and start == 0:
        if update == 2 or update == num_updates or update == (num_updates // 10):
            return True
        if num_updates >= 100 and update % (num_updates // 100) == 0:
            return True
    return False

def dormant_neurons(activation_values, plot_inter):
    threshold = 0.2
    num_neurons = 0
    num_dormant = 0
    all_scores = []

    for layer_name, activations in activation_values.items():
        if isinstance(layer_name, nn.Tanh):
            abs_activations = torch.abs(activations)
            mean_per_neuron = abs_activations.mean(dim=0)
            mean_layer = mean_per_neuron.mean()
            score = mean_per_neuron / (mean_layer + 1e-8)

            all_scores.extend(score)
            num_neurons += score.numel()
            num_dormant += (score <= threshold).sum().item()

    # if plot_inter % 2000 == 0:
    #     plt.hist(all_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    #     plt.xlabel("Score")
    #     plt.ylabel("# Neurons")
    #     plt.title("Distribution")
    #     plt.axvline(threshold, color='red', linestyle='dashed', label="Threshold = {}".format(threshold))
    #     plt.legend()
    #     plt.show()
    #     plt.close()

    return round((100 * num_dormant / num_neurons), 4)

def weight_magnitude(weight_dict):
    absolute_weights = [w.abs() for w in weight_dict.values()]
    total_absolute_value = sum(w.sum() for w in absolute_weights)
    total_weight_count = sum(w.numel() for w in absolute_weights)

    return round((total_absolute_value / (total_weight_count + 1e-8)).item(), 4)

def effective_rank(weight_dict, hidden_size):
    W = weight_dict[list(weight_dict.keys())[-2]]
    U, S, Vh = torch.linalg.svd(W)
    S = S / S.sum()
    entropy = -torch.sum(S * torch.log(S + 1e-8))
    return round(torch.exp(entropy).item() / hidden_size, 4)

def gradient_monitoring(results_dir, update, num_updates, flag_grad, grad_):
    length = min(64, grad_.size(0))
    grad = grad_.narrow(0, 0, length).detach().cpu().numpy()

    dot_product = grad @ grad.T
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    dot_product_norm = dot_product / (norms @ norms.T + 1e-8)

    covariance_matrix = np.cov(grad)

    if update == 2 or update == (num_updates // 10) or update == num_updates:
        if update == 2:
            record_index = "training progress 0%"
        elif update == (num_updates // 10):
            record_index = "training progress 10%"
        else:
            record_index = "training progress 100%"

        cmap = sns.cubehelix_palette(n_colors=50, rot=-0.5, gamma=1, light=0.95)
        custom_cmap = LinearSegmentedColormap.from_list(
            'orange_white_green',
            ['#f78b6c', 'white', '#96b2c9']
        )

        if flag_grad == 0:

            plt.figure(figsize=(10, 8))
            sns.heatmap(dot_product_norm, annot=False, cmap=custom_cmap, vmin=-1.0, vmax=1.0, cbar=True)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(results_dir, "dot product ({}).png".format(record_index)), dpi=300)
            plt.show()
            plt.close()

            # plt.figure(figsize=(10, 8))
            # sns.heatmap(covariance_matrix, annot=False, cmap=cmap, cbar=False)
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig(os.path.join(results_dir, "gradient covariance ({}).png".format(record_index)), dpi=300)
            # plt.show()
            # plt.close()

    return flag_grad + 1, round(dot_product_norm.mean().item(), 4), round(covariance_matrix.mean().item(), 4)

def hessian_vector_product(loss, model, v):
    grads = grad(loss, model.parameters(), create_graph=True)
    grad_vector = torch.cat([g.view(-1) for g in grads])
    hvp = grad(grad_vector @ v, model.parameters(), retain_graph=True)
    hvp_vector = torch.cat([h.contiguous().view(-1) for h in hvp])
    return hvp_vector

def loss_landscape_sharpness(model, loss, num_iters=20):
    # Compute the largest hessian eigval

    # Flatten all parameters
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)

    # Random initial vector
    v = torch.randn(total_params).to(next(model.parameters()).device)
    v = v / torch.norm(v)

    for _ in range(num_iters):
        hv = hessian_vector_product(loss, model, v)
        eigval = torch.dot(hv, v)
        v = hv / (torch.norm(hv) + 1e-8)

    return round(eigval.item(), 4)