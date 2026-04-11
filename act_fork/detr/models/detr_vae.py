# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR-VAE model for ACT (Action Chunking with Transformers).
Fork modified for state-only operation with configurable dimensions.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """DETR-VAE for ACT. Supports both vision and state-only modes."""
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim,
                 num_queries, camera_names):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # State-only path
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.backbones = None

        # CVAE encoder parameters
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))

        # Decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, state_dim
        image: batch, num_cam, C, H, W  (or None for state-only)
        env_state: None (unused)
        actions: batch, seq, action_dim  (None at inference)
        is_pad: batch, seq  (None at inference)
        """
        is_training = actions is not None
        bs, _ = qpos.shape

        ### Obtain latent z from action sequence
        if is_training:
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+2, bs, hidden_dim)
            # padding mask
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
            # position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
            # CVAE encoder forward
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take CLS output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        ### Policy decoder
        if self.backbones is not None:
            # Image observation path (unchanged from original)
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos,
                                  latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            # State-only path: build [latent, qpos] as 2-token sequence
            proprio_input = self.input_proj_robot_state(qpos)  # (bs, hidden_dim)
            src = torch.stack([latent_input, proprio_input], dim=1)  # (bs, 2, hidden_dim)
            hs = self.transformer(src, None, self.query_embed.weight,
                                  self.additional_pos_embed.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(args):
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = getattr(args, 'state_dim', 14)
    action_dim = getattr(args, 'action_dim', state_dim)

    if getattr(args, 'state_only', False):
        backbones = None
    else:
        from .backbone import build_backbone
        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

    transformer = build_transformer(args)
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
