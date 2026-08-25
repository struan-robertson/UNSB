"""How many styles per shoeprint cover a generator's output diversity?

Motivates the size of the pre-generated shoemark pool used to train the
siamese network (in-loop generation is too slow for UNSB's 5-NFE sampler, and
for consistency the GAN row is pooled the same way). Run once per campaign
row on its selected checkpoint: the coverage criterion is the fixed part of
the protocol, and N is wherever that run's style distribution meets it.

Criterion: for a pool of N random styles, coverage C(N) is the mean distance
from a fresh random-style render to its nearest pool member (InceptionV3 pool
features, cosine distance, matching the FID/CIS feature convention; per-image
min-max normalisation). For UNSB the bridge is stochastic — re-rendering the
*same* style draws fresh trajectory noise — so the paired same-style distance
is the generator's resolution floor: once C(N) reaches it, additional styles
are indistinguishable from trajectory noise. The GAN is deterministic given a
style (floor 0), so its recommendation uses the 95%-of-achievable-reduction
point of the curve instead.

Writes style_analysis/style_coverage_<backend>.{csv,png} and prints the
recommendation. Run from the siamese venv (has matplotlib and both backends;
GPU must be free):

    cd ~/Development/Doctorate/siamese && uv run python \
        ~/Development/Doctorate/diffusion/UNSB/style_coverage.py --backend unsb
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unsb_handler import GeneratorHandler, load_unsb_opt

PRINT_DIR = Path('~/Vault/University/Doctorate/Data/Siamese/Compiled/Shoeprints/train').expanduser()
GAN_CONFIG = Path('/home/struan/Development/Doctorate/one_to_many_gan/implementation/config.toml')
# the adopted bridge is deterministic, so that is the default; bridge_noise must
# match training, and the stochastic ablations need an explicit config.toml
UNSB_CONFIG = Path(__file__).resolve().parent / 'config_no_noise.toml'
MUNIT_REPO = Path('~/Development/Doctorate/MUNIT').expanduser()
OUT_DIR = Path(__file__).resolve().parent / 'style_analysis'
N_PRINTS = 20      # shoeprints analysed (seeded random draw from the train set)
M = 256            # candidate random styles rendered per print
K_NOISE = 16       # same-style render pairs measuring the noise floor (unsb only)
POOL_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
R = 20             # random pool draws averaged per size
BATCH = 16


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


def load_print(path):
    img = Image.open(path).convert('L')
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]
    if x.shape[-2:] != (512, 256):
        x = F.interpolate(x[None], (512, 256), mode='bicubic', align_corners=False,
                          antialias=True)[0].clamp(0, 1)
    return x


def render(handler, device, print_img, n, styles=None, batch=BATCH):
    """n renders of one print: explicit z styles (unsb) or fresh internal ones."""
    outs = []
    for i in range(0, n, batch):
        bs = min(batch, n - i)
        src = print_img[None].expand(bs, -1, -1, -1).to(device)
        z = styles[i:i + bs].to(device) if styles is not None else None
        outs.append(handler.generate(src, style=z, difficulty=1.0))
    return torch.cat(outs)


@torch.no_grad()
def embed(images, inception, device, batch=BATCH):
    """InceptionV3 pool features of [-1,1] 1-channel renders, L2-normalised."""
    feats = []
    for x in images.split(batch):
        x = F.interpolate(x.to(device), (299, 299), mode='bicubic',
                          align_corners=False, antialias=True)
        x = x.expand(-1, 3, -1, -1)
        lo = x.amin(dim=(1, 2, 3), keepdim=True)
        hi = x.amax(dim=(1, 2, 3), keepdim=True)
        x = (x - lo) / (hi - lo + 1e-14)
        feats.append(inception(x).cpu())
    return F.normalize(torch.cat(feats), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['unsb', 'gan', 'munit'], default='unsb')
    parser.add_argument('--generator-config', type=Path, default=None,
                        help='backend config overriding the built-in default (unsb: %s, '
                             'so the stochastic-bridge ablations need config.toml)'
                             % UNSB_CONFIG.name)
    parser.add_argument('--print-dir', type=Path, default=PRINT_DIR,
                        help='shoeprint train directory (default: the local Vault path)')
    parser.add_argument('--tag', default=None,
                        help='label for the output files (default: the backend name); '
                             'per-run campaign invocations pass the run name so '
                             'analyses do not overwrite each other')
    parser.add_argument('--nfe', type=int, default=None,
                        help='override the bridge step count for this measurement '
                             '(unsb only); appends _nfe<N> to the tag, so a sweep '
                             'cannot overwrite the trained-count measurement')
    parser.add_argument('--batch', type=int, default=BATCH,
                        help='render and Inception batch size; lower it to share a GPU '
                             '(keep <=16 on ROCm, where MIOpen falls off its fast path)')
    parser.add_argument('--m', type=int, default=M,
                        help='candidate random styles rendered per shoeprint; raise it '
                             'when N_95 is unreached, which happens for a deterministic '
                             'generator (zero floor) whose 5%% target is the harshest')
    parser.add_argument('--pool-sizes', type=int, nargs='+', default=POOL_SIZES,
                        help='pool sizes at which C(N) is measured; the largest must '
                             'leave enough held-out styles to average over, so raise --m '
                             'alongside it')
    args = parser.parse_args()
    if args.backend == 'munit' and args.generator_config is None:
        parser.error('the munit backend needs --generator-config (a training yaml '
                     'with a checkpoint: key)')
    if max(args.pool_sizes) >= args.m:
        parser.error('--pool-sizes must stay below --m (%d): a pool of every candidate '
                     'style leaves nothing held out to measure coverage against' % args.m)
    pool_sizes = sorted(args.pool_sizes)
    tag = args.tag or args.backend
    if args.nfe is not None:
        tag = '%s_nfe%d' % (tag, args.nfe)
    if args.m != M:
        # a denser grid supersedes rather than replaces: keep both measurements
        # on disk so the change in M can be checked against the coarser run
        tag = '%s_m%d' % (tag, args.m)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    handler = build_handler(args.backend, device, args.generator_config)
    if args.nfe is not None:
        if args.backend != 'unsb':
            parser.error('--nfe applies to the unsb backend only')
        handler.opt.num_timesteps = args.nfe

    from torchvision.models.inception import inception_v3
    inception = inception_v3(weights='DEFAULT').to(device).eval()
    inception.fc = torch.nn.Identity()

    print_dir = args.print_dir.expanduser()
    files = sorted(print_dir.rglob('*.png')) + sorted(print_dir.rglob('*.jpg'))
    gen = torch.Generator().manual_seed(0)
    picks = [files[i] for i in torch.randperm(len(files), generator=gen)[:N_PRINTS]]

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    style_dim = handler.opt.style_dim if args.backend == 'unsb' else None
    curves, floors = [], []
    with torch.no_grad():
        for path in tqdm(picks, desc='prints', unit='print'):
            print_img = load_print(path)

            zs = torch.randn(args.m, style_dim) if style_dim else None
            f = embed(render(handler, device, print_img, args.m, zs, batch=args.batch),
                      inception, device, batch=args.batch)

            if args.backend == 'unsb':
                # same styles re-rendered: the trajectory-noise resolution floor
                z_pairs = torch.randn(K_NOISE, style_dim).repeat_interleave(2, dim=0)
                f_pairs = embed(render(handler, device, print_img, 2 * K_NOISE, z_pairs,
                                       batch=args.batch),
                                inception, device, batch=args.batch)
                floors.append(float((1 - (f_pairs[0::2] * f_pairs[1::2]).sum(1)).mean()))

            dist = 1 - f @ f.T  # [m, m] cosine distances
            curve = []
            for n in pool_sizes:
                cov = []
                for _ in range(R):
                    pool = rng.choice(args.m, size=n, replace=False)
                    held = np.setdiff1d(np.arange(args.m), pool)
                    cov.append(float(dist[held][:, pool].min(dim=1).values.mean()))
                curve.append(np.mean(cov))
            curves.append(curve)

    curves = np.array(curves)          # [N_PRINTS, len(pool_sizes)]
    mean_c, std_c = curves.mean(0), curves.std(0)
    floor = float(np.mean(floors)) if floors else 0.0

    # saturation: within 10% of the floor (unsb), or 95% of achievable reduction.
    # The 95% point is log-log interpolated between the bracketing measured
    # sizes so every run is sized at the same coverage level, not a grid point
    rel = (mean_c - floor) / max(mean_c[0] - floor, 1e-12)
    at_floor = [n for n, c in zip(pool_sizes, mean_c) if c <= 1.1 * floor]
    n_95 = None
    idx = next((i for i, r in enumerate(rel) if r <= 0.05), None)
    if idx == 0:
        n_95 = pool_sizes[0]
    elif idx is not None:
        n0, n1 = pool_sizes[idx - 1], pool_sizes[idx]
        r0, r1 = rel[idx - 1], max(rel[idx], 1e-12)
        t = (np.log(0.05) - np.log(r0)) / (np.log(r1) - np.log(r0))
        n_95 = int(np.ceil(np.exp(np.log(n0) + t * (np.log(n1) - np.log(n0)))))

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / ('style_coverage_%s.csv' % tag)
    with csv_path.open('w') as fh:
        fh.write('pool_size,coverage_mean,coverage_std,rel_residual\n')
        for n, c, s, r in zip(pool_sizes, mean_c, std_c, rel):
            fh.write('%d,%.6f,%.6f,%.4f\n' % (n, c, s, r))
        fh.write('# noise_floor,%.6f\n' % floor)
        # the two statistics the thesis tables quote, so they need not be
        # re-derived from the curve by hand for every row
        fh.write('# c1,%.6f\n' % mean_c[0])
        fh.write('# n_95,%s\n' % (n_95 if n_95 is not None else 'unreached'))
        fh.write('# nfe,%s\n' % (args.nfe if args.nfe is not None
                                 else getattr(handler.opt, 'num_timesteps', 'n/a')))
        # an unreached n_95 is a property of the grid as much as of the model,
        # so the grid that produced it belongs with the number
        fh.write('# m,%d\n' % args.m)

    print('\n[%s] pool N | coverage C(N) | residual vs floor' % args.backend)
    for n, c, r in zip(pool_sizes, mean_c, rel):
        print('%6d | %.4f        | %5.1f%%' % (n, c, 100 * r))
    if args.backend == 'unsb':
        print('trajectory-noise floor: %.4f' % floor)
        print('N at <=1.1x floor: %s'
              % (at_floor[0] if at_floor else 'not reached by %d' % pool_sizes[-1]))
    else:
        print('deterministic generator: floor criterion n/a')
    print('N at 95%% coverage: %s'
          % (n_95 if n_95 is not None
             else 'not reached by %d — raise M and pool_sizes' % pool_sizes[-1]))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(pool_sizes, mean_c, yerr=std_c, marker='o', capsize=3,
                    label='coverage C(N)')
        if floor > 0:
            ax.axhline(floor, ls='--', c='grey', label='trajectory-noise floor')
        ax.set_xscale('log', base=2)
        ax.set_xlabel('styles per shoeprint (pool size N)')
        ax.set_ylabel('mean nearest-pool cosine distance')
        ax.set_title(args.backend)
        ax.legend()
        fig.tight_layout()
        out = OUT_DIR / ('style_coverage_%s.png' % tag)
        fig.savefig(out, dpi=150)
        print('wrote %s' % out)
    except ImportError:
        print('matplotlib unavailable; skipped the plot')


if __name__ == '__main__':
    main()