"""Quick sanity test for the state-only ACT fork."""
import sys
sys.path.insert(0, '/home/claude/act_fork')

import torch
import numpy as np
import h5py
import os
import tempfile

# Test 1: Model builds correctly
print("=" * 60)
print("Test 1: Build state-only ACT model")
from detr.main import build_ACT_model_and_optimizer

config = {
    'lr': 1e-4,
    'num_queries': 20,
    'kl_weight': 10,
    'hidden_dim': 256,
    'dim_feedforward': 1024,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': [],
    'state_only': True,
    'state_dim': 30,
    'action_dim': 9,
}

model, optimizer = build_ACT_model_and_optimizer(config)
print(f"  Model on cuda: {next(model.parameters()).is_cuda}")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {n_params:,}")

# Test 2: Forward pass (training)
print("\nTest 2: Training forward pass")
bs = 4
qpos = torch.randn(bs, 30).cuda()
actions = torch.randn(bs, 20, 9).cuda()
is_pad = torch.zeros(bs, 20).bool().cuda()

a_hat, is_pad_hat, (mu, logvar) = model(qpos, None, None, actions, is_pad)
print(f"  a_hat shape: {a_hat.shape}")         # should be (1, bs, 20, 9)
print(f"  mu shape: {mu.shape}")               # should be (bs, 32)
print(f"  logvar shape: {logvar.shape}")        # should be (bs, 32)

# Test 3: Forward pass (inference)
print("\nTest 3: Inference forward pass")
a_hat, _, (mu, logvar) = model(qpos, None, None)
print(f"  a_hat shape: {a_hat.shape}")
print(f"  mu is None: {mu is None}")

# Test 4: Full policy wrapper
print("\nTest 4: ACTPolicy wrapper")
from policy import ACTPolicy

policy = ACTPolicy(config)
policy.cuda()

# Training
policy.train()
loss_dict = policy(qpos, image=None, actions=actions, is_pad=is_pad)
print(f"  Loss dict: l1={loss_dict['l1'].item():.5f}, kl={loss_dict['kl'].item():.5f}, loss={loss_dict['loss'].item():.5f}")

# Inference
policy.eval()
with torch.no_grad():
    pred_actions = policy(qpos)
print(f"  Inference pred_actions shape: {pred_actions.shape}")

# Test 5: Data loading
print("\nTest 5: Data loading with dummy HDF5")
tmpdir = tempfile.mkdtemp()
for i in range(5):
    fpath = os.path.join(tmpdir, f"teleop_skill2_{i:03d}.hdf5")
    with h5py.File(fpath, "w") as f:
        T = np.random.randint(50, 100)
        ep = f.create_group("episode_0")
        ep.create_dataset("obs", data=np.random.randn(T, 30).astype(np.float32))
        ep.create_dataset("actions", data=np.random.randn(T, 9).astype(np.float32))
        ep.create_dataset("teleop_active", data=np.ones(T, dtype=bool))

from utils import load_data
train_dl, val_dl, norm_stats = load_data(
    demo_dir=tmpdir,
    skill="approach_and_grasp",
    chunk_size=20,
    batch_size_train=4,
    batch_size_val=2,
)
print(f"  Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")
print(f"  Norm stats keys: {list(norm_stats.keys())}")

# Test 6: One training step
print("\nTest 6: One full training step")
policy.train()
optimizer = policy.configure_optimizers()
for batch in train_dl:
    qpos_data, action_data, is_pad = batch
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()

    forward_dict = policy(qpos_data, image=None, actions=action_data, is_pad=is_pad)
    loss = forward_dict['loss']
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  Step loss: {loss.item():.5f}")
    break

# Test 7: AMP training step
print("\nTest 7: AMP (bf16) training step")
scaler = torch.amp.GradScaler('cuda')
for batch in train_dl:
    qpos_data, action_data, is_pad = batch
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        forward_dict = policy(qpos_data, image=None, actions=action_data, is_pad=is_pad)
        loss = forward_dict['loss']

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    print(f"  AMP Step loss: {loss.item():.5f}")
    break

# Cleanup
import shutil
shutil.rmtree(tmpdir)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
