"""
State-only ACT BC Training Script.
Fork of official ACT imitate_episodes.py, simplified for state-only operation.

Usage:
    python imitate_episodes.py \
        --demo_dir /path/to/demos \
        --skill approach_and_grasp \
        --ckpt_dir checkpoints/act_skill2 \
        --chunk_size 20 \
        --batch_size 32 \
        --num_epochs 3000 \
        --lr 1e-4 \
        --kl_weight 10 \
        --hidden_dim 256 \
        --dim_feedforward 1024
"""
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from utils import load_data, compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy


def main(args):
    set_seed(args['seed'])

    ckpt_dir = args['ckpt_dir']
    num_epochs = args['num_epochs']
    is_eval = args.get('eval', False)

    if is_eval:
        print("Evaluation mode — use eval_act_bc.py for Isaac Sim evaluation.")
        return

    # ---- Load data ----
    train_dataloader, val_dataloader, norm_stats = load_data(
        demo_dir=args['demo_dir'],
        skill=args['skill'],
        chunk_size=args['chunk_size'],
        batch_size_train=args['batch_size'],
        batch_size_val=args['batch_size'],
        filter_active=args.get('filter_active', True),
    )

    # Determine dims from data
    sample_qpos, sample_action, _ = next(iter(train_dataloader))
    state_dim = sample_qpos.shape[-1]
    action_dim = sample_action.shape[-1]
    print(f"[Config] state_dim={state_dim}, action_dim={action_dim}")

    # ---- Build policy ----
    policy_config = {
        'lr': args['lr'],
        'num_queries': args['chunk_size'],
        'kl_weight': args['kl_weight'],
        'hidden_dim': args['hidden_dim'],
        'dim_feedforward': args['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': args.get('enc_layers', 4),
        'dec_layers': args.get('dec_layers', 7),
        'nheads': args.get('nheads', 8),
        'camera_names': [],
        'state_only': True,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'action_loss_weights': args.get('action_loss_weights', None),
    }

    # ---- Save config and stats ----
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'obs_mean': norm_stats['obs_mean'].numpy(),
            'obs_std': norm_stats['obs_std'].numpy(),
            'action_mean': norm_stats['action_mean'].numpy(),
            'action_std': norm_stats['action_std'].numpy(),
        }, f)
    print(f"[Config] Saved dataset_stats.pkl to {ckpt_dir}")

    config_path = os.path.join(ckpt_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump({**args, **policy_config, 'state_dim': state_dim, 'action_dim': action_dim}, f)

    # ---- Train ----
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, policy_config, num_epochs, ckpt_dir, args['seed'])
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def forward_pass(data, policy):
    """Forward pass for state-only ACT."""
    qpos_data, action_data, is_pad = data
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    # State-only: no image
    return policy(qpos_data, image=None, actions=action_data, is_pad=is_pad)


def train_bc(train_dataloader, val_dataloader, policy_config, num_epochs, ckpt_dir, seed):
    set_seed(seed)

    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    # AMP for faster training on A100
    scaler = torch.amp.GradScaler('cuda')
    use_amp = True

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        # ---- Validation ----
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        # ---- Logging ----
        if epoch % 50 == 0 or epoch == 0:
            summary_string = ' '.join(f'{k}: {v.item():.5f}' for k, v in epoch_summary.items())
            print(f'\nEpoch {epoch} | Val: {summary_string}')

        # ---- Training ----
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    forward_dict = forward_pass(data, policy)
                    loss = forward_dict['loss']
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                forward_dict = forward_pass(data, policy)
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        # ---- Train epoch summary ----
        if epoch % 50 == 0 or epoch == 0:
            n_batches = batch_idx + 1
            epoch_summary = compute_dict_mean(train_history[-n_batches:])
            summary_string = ' '.join(f'{k}: {v.item():.5f}' for k, v in epoch_summary.items())
            print(f'         Train: {summary_string}')

        # ---- Periodic checkpoint ----
        if epoch % 500 == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # ---- Final save ----
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished: Seed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train', alpha=0.5)
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('State-only ACT BC Training')

    # Data
    parser.add_argument('--demo_dir', type=str, required=True,
                        help='Directory containing HDF5 demo files')
    parser.add_argument('--skill', type=str, default='approach_and_grasp',
                        choices=['approach_and_grasp', 'carry_and_place', 'navigate'])
    parser.add_argument('--filter_active', action='store_true', default=True,
                        help='Filter by teleop_active flag')

    # Save
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory to save checkpoints')

    # Model
    parser.add_argument('--chunk_size', type=int, default=20,
                        help='Action chunk size (num_queries)')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=7)
    parser.add_argument('--nheads', type=int, default=8)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kl_weight', type=float, default=10.0)
    parser.add_argument('--action_loss_weights', type=float, nargs='+', default=None,
                        help='Per-dimension loss weights for actions. '
                             'E.g. for 9D: 1 1 1 1 1 1 1 10 10 to upweight wheels')
    parser.add_argument('--seed', type=int, default=0)

    main(vars(parser.parse_args()))
