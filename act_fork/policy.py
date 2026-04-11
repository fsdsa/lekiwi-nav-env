"""
ACT Policy wrapper.
Modified for state-only operation (no image input).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from detr.main import build_ACT_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # DETRVAE
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.state_only = args_override.get('state_only', False)

        # Per-dimension action loss weights (amplify sparse dims like wheels)
        action_dim = args_override.get('action_dim', 9)
        action_loss_weights = args_override.get('action_loss_weights', None)
        if action_loss_weights is not None:
            self.register_buffer('action_weights',
                                 torch.tensor(action_loss_weights, dtype=torch.float32))
        else:
            self.register_buffer('action_weights', torch.ones(action_dim, dtype=torch.float32))

        print(f'KL Weight {self.kl_weight}, State-only: {self.state_only}')
        if action_loss_weights is not None:
            print(f'Action loss weights: {action_loss_weights}')

    def __call__(self, qpos, image=None, actions=None, is_pad=None):
        env_state = None

        if not self.state_only and image is not None:
            # Vision mode: normalize images
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            image = normalize(image)

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # (B, K, action_dim)
            # Apply per-dimension weights
            weighted_l1 = all_l1 * self.action_weights.unsqueeze(0).unsqueeze(0)  # broadcast
            l1 = (weighted_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
