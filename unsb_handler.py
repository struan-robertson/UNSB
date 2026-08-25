"""Drop-in generator handler for the siamese training pipeline.

Mirrors the interface of one_to_many_gan.GeneratorHandler so the siamese
project can generate synthetic shoemarks with the trained UNSB model instead
of the GAN: 1-channel [0, 1] shoeprints in, 1-channel tanh-range [-1, 1]
shoemarks out (identical to the GAN's convention).

The UNSB repo is a flat script-style package (not installable), so callers
add the repo root to sys.path and import from there:

    sys.path.insert(0, str(unsb_repo))
    from unsb_handler import GeneratorHandler, load_unsb_opt

    opt = load_unsb_opt(unsb_repo / 'config.toml')
    handler = GeneratorHandler(opt, device)
    shoemarks = handler.generate(shoeprints)  # [B, 1, H, W] in [0, 1]

The checkpoint loaded is checkpoints/<name>/<epoch>_net_G.pth, with <name>
from [experiment] and <epoch> from [test] in the config file.
"""
import os
from pathlib import Path

import torch

from models import networks
from options.test_options import TestOptions
from util.evaluation import sb_translate


def load_unsb_opt(config_path):
    """Build the UNSB options namespace from a config.toml, without CLI parsing.

    Mirrors BaseOptions.parse() minus its side effects (writing opt.txt to a
    cwd-relative path, torch.cuda.set_device), and resolves checkpoints_dir
    relative to the config file so it works from any working directory.
    """
    config_path = Path(config_path).expanduser().resolve()
    opts = TestOptions(cmd_line='--config %s' % config_path)
    opt = opts.gather_options()
    opt.isTrain = False
    if opt.load_size_h is None:
        opt.load_size_h = opt.load_size
    if opt.crop_size_h is None:
        opt.crop_size_h = opt.crop_size
    opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(',') if int(i) >= 0]
    if not os.path.isabs(opt.checkpoints_dir):
        opt.checkpoints_dir = str(config_path.parent / opt.checkpoints_dir)
    return opt


class GeneratorHandler:
    """Generate shoemark images from shoeprints with a trained UNSB checkpoint."""

    def __init__(self, opt, device):
        opt.gpu_ids = [device.index] if device.type == 'cuda' else []

        generator = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                      opt.gpu_ids, opt)
        ckpt = Path(opt.checkpoints_dir) / opt.name / ('%s_net_G.pth' % opt.epoch)
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
        generator.load_state_dict(state_dict)
        generator.to(device).eval()

        for param in generator.parameters():
            param.requires_grad = False

        self.device = device
        self.opt = opt
        self.generator = generator
        # MIOpen (ROCm) settles on ~6x-slower conv kernels for batches above
        # ~24 at 512x256 on the 7900 XTX; small fixed chunks stay on the fast
        # path and bound its one-off per-shape kernel search to a few shapes.
        # cudnn has no such cliff, so CUDA builds run unchunked
        self.max_chunk = 16 if torch.version.hip else None

    def generate(
        self,
        shoeprints: torch.Tensor,
        *,
        normalised=False,
        difficulty: float | None = None,
        style: torch.Tensor | None = None,
    ):
        """Translate a batch of shoeprints through the full 5-step bridge.

        difficulty is accepted for call-site compatibility with the GAN handler
        but ignored: UNSB has no trained domain variable, and the curriculum it
        drove was empirically inert. style=None samples a fresh z per sample.
        """
        if not normalised:
            shoeprints = shoeprints * 2.0 - 1.0  # [0, 1] -> the model's [-1, 1]

        chunk = self.max_chunk or shoeprints.shape[0]
        chunks = shoeprints.split(chunk)
        styles = style.split(chunk) if style is not None else [None] * len(chunks)
        return torch.cat([sb_translate(self.generator, c, self.opt, style=s)
                          for c, s in zip(chunks, styles)])
