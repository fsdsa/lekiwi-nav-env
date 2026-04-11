# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Build ACT model and optimizer.
Modified to support state-only mode without requiring command-line args.
"""
import argparse
import torch
from .models import build_ACT_model


def get_args_parser():
    parser = argparse.ArgumentParser('ACT model config', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # Backbone
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--camera_names', default=[], type=list)

    # Transformer
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=7, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # State-only extensions
    parser.add_argument('--state_only', action='store_true')
    parser.add_argument('--state_dim', default=14, type=int)
    parser.add_argument('--action_dim', default=14, type=int)

    return parser


def build_ACT_model_and_optimizer(args_override):
    """Build model from a dict of config overrides (no command-line parsing)."""
    parser = get_args_parser()
    # Parse with empty args to get defaults
    args = parser.parse_args([])

    # Override with provided config
    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer
