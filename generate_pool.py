"""Pre-generate a pool of N synthetic shoemarks per shoeprint class.

Replaces in-loop generation in the siamese pipeline: marks are generated once
here and streamed from disk by the siamese StreamingDataset (UNSB's 5-NFE
sampling is too slow in-loop, and the GAN row is pooled identically for
consistency). Pick N per run with style_coverage.py on the run's selected
checkpoint (fixed coverage criterion, per-run N).

Backends: 'unsb' loads checkpoint [experiment].name + [test].epoch from the
UNSB config_no_noise.toml (the adopted deterministic bridge); 'gan' loads
[inference].checkpoint from the one_to_many_gan config.toml — set each to the
best-FID checkpoint before running.

Layout: <out-dir>/<class_id>/<k>.png, one directory per shoeprint class,
matching the class ids of the siamese dataset (get_id convention). Classes
with several print images rotate through them. Images are saved in [0, 1]
(white background); the siamese pipeline maps them back to the generator's
[-1, 1] when substituting them for in-loop generation. Seeded and idempotent:
existing files are kept, so an interrupted run resumes where it left off.

Run from the siamese venv (it has both backends importable):

    cd ~/Development/Doctorate/siamese && uv run python \
        ~/Development/Doctorate/diffusion/UNSB/generate_pool.py \
        --backend unsb --n-styles 32
"""
import argparse
import sys
from pathlib import Path

import torch
from impression_tools.pool import find_prints, generate_pool, outstanding

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unsb_handler import GeneratorHandler, load_unsb_opt

PRINT_DIR = Path('~/Vault/University/Doctorate/Data/Siamese/Compiled/Shoeprints/train').expanduser()
GAN_CONFIG = Path('/home/struan/Development/Doctorate/one_to_many_gan/implementation/config.toml')
# the adopted bridge is deterministic, so that is the default; bridge_noise must
# match training, and the stochastic ablations need an explicit config.toml
UNSB_CONFIG = Path(__file__).resolve().parent / 'config_no_noise.toml'
MUNIT_REPO = Path('~/Development/Doctorate/MUNIT').expanduser()
# bulk working storage (~/Extra), deliberately away from the curated ~/Vault data
OUT_DIRS = {
    'unsb': Path('~/Extra/Doctorate/synthetic/Shoemarks_UNSB/train'),
    'gan': Path('~/Extra/Doctorate/synthetic/Shoemarks_GAN/train'),
    'munit': Path('~/Extra/Doctorate/synthetic/Shoemarks_MUNIT/train'),
}


def build_handler(backend, device, config=None):
    if backend == 'unsb':
        return GeneratorHandler(load_unsb_opt(config or UNSB_CONFIG), device)
    if backend == 'munit':
        # spec-loaded: the MUNIT repo's top-level data.py would collide with
        # this repo's data package if its root joined sys.path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'munit_handler', MUNIT_REPO / 'munit_handler.py')
        munit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(munit)
        # munit configs carry their own checkpoint: key, so config is required
        return munit.GeneratorHandler(munit.load_munit_config(config), device)
    from one_to_many_gan import GeneratorHandler as GanHandler, load_gan_config
    return GanHandler(load_gan_config(config or GAN_CONFIG), device)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['unsb', 'gan', 'munit'], default='unsb')
    parser.add_argument('--n-styles', type=int, required=True,
                        help='marks generated per shoeprint class (from style_coverage.py)')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='default: Shoemarks_<BACKEND>/train next to the real shoemarks')
    parser.add_argument('--difficulty', type=float, default=1.0,
                        help="GAN domain variable; values below 1.0 write to <out-dir>_d<value> "
                             "sibling trees for the pooled curriculum (gan backend only)")
    parser.add_argument('--generator-config', type=Path, default=None,
                        help="backend config overriding the built-in default (unsb: %s, so "
                             "the stochastic-bridge ablations need config.toml); campaign "
                             "rows point this at per-checkpoint configs" % UNSB_CONFIG.name)
    parser.add_argument('--print-dir', type=Path, default=PRINT_DIR,
                        help='shoeprint train directory (default: the local Vault path)')
    parser.add_argument('--frozen-style', action='store_true',
                        help='one seeded style per shoeprint class, reused for every render, '
                             'so only trajectory noise varies (unsb backend only)')
    args = parser.parse_args()
    if args.difficulty != 1.0 and args.backend != 'gan':
        parser.error('--difficulty only applies to the gan backend (UNSB ignores it)')
    if args.frozen_style and args.backend != 'unsb':
        parser.error('--frozen-style only applies to the unsb backend')
    if args.backend == 'munit' and args.generator_config is None:
        parser.error('the munit backend needs --generator-config (a training yaml '
                     'with a checkpoint: key)')
    out_root = (args.out_dir or OUT_DIRS[args.backend]).expanduser()
    if args.difficulty != 1.0:
        # matches the sibling-tree naming the siamese StreamingDataset discovers
        out_root = out_root.with_name('%s_d%g' % (out_root.name, args.difficulty))

    print_dir = args.print_dir.expanduser()
    classes = find_prints(print_dir)
    print('%d classes, %d marks each -> %d images -> %s'
          % (len(classes), args.n_styles, len(classes) * args.n_styles, out_root))

    todo = outstanding(classes, out_root, args.n_styles)
    print('%d images to generate (%d already on disk)'
          % (len(todo), len(classes) * args.n_styles - len(todo)))
    if not todo:
        # a complete (e.g. transferred) pool needs no generator: don't require
        # this run's checkpoint on the machine just to no-op over it
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    handler = build_handler(args.backend, device, args.generator_config)

    # per-class styles drawn from their own generator in sorted-class order:
    # stable across resumes regardless of which files already exist
    styles = None
    if args.frozen_style:
        g = torch.Generator().manual_seed(0)
        styles = {cid: torch.randn(handler.opt.style_dim, generator=g)
                  for cid in sorted(classes)}

    generate_pool(handler, todo, device, difficulty=args.difficulty, styles=styles)


if __name__ == '__main__':
    main()