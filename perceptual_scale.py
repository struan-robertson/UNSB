"""What does a coverage distance of 0.007 actually look like?

The coverage criterion of style_coverage.py counts a style as distinguishable
while the nearest-pool distance stays above a threshold, and for a
deterministic generator (the GAN of Chapter 5, and the bridge trained with
bridge_noise = 0) that threshold is 5% of C(1) with no noise floor to stop it.
That places N_95 wherever the curve happens to reach a cosine distance of
roughly 0.007 in InceptionV3 feature space, and nothing in the metric says
whether two renders that far apart differ in any way a person could see.

This measures the answer. One shoeprint, M candidate styles, and for each
target distance K pairs of renders sampled from those separated by roughly
that much; the targets are the measured C(N) of that run's coverage curve, so
every column is labelled with the pool size it justifies. LPIPS is reported as
a mean over the K pairs with its standard error, and had no part in selecting
them, so a level where the cosine distance keeps falling while LPIPS does not
is a level at which the coverage metric is counting variation a perceptual
metric cannot see.

One column comes from a different model: the *stochastic* full model rendering
one style twice, whose separation is trajectory noise alone. A deterministic
generator has no floor, so nothing internal to it says whether 0.007 is a lot
or a little; that column is a borrowed yardstick, the one distance in the
figure whose meaning is known by construction ("the same mark with different
speckle"). Its position in the ordering is the result: style levels lying to
the right of it are separated by less than pure noise separates two renders.

The difference row uses one fixed gain across every column, so a column that
fades to white is one whose renders differ less than its neighbours' do,
rather than one that has been normalised to look busy.

    perceptual_scale.py --generator-config campaign_unsb_no_noise_1.toml \
        --coverage-csv style_analysis/style_coverage_unsb_no_noise_1_m1024.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from style_coverage import build_handler, embed, load_print, render  # noqa: E402
import style_space_unsb as ss  # noqa: E402

OUT_DIR = REPO / 'style_analysis' / 'perceptual_scale'
M = 512        # candidate styles the pairs are drawn from
K = 16         # pairs averaged per level
LEVELS = [1, 64, 192, 384, 768]
DIFF_GAIN = 6  # shared by every difference panel; see the module docstring
REF_LABEL = 'full model\none style, two trajectories'


def read_coverage(path):
    """The measured curve and C(1) from a style_coverage.py CSV."""
    curve, c1 = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('# c1,'):
            c1 = float(line.split(',')[1])
        elif line and not line.startswith(('#', 'pool_size')):
            n, mean = line.split(',')[:2]
            curve[int(n)] = float(mean)
    if c1 is None:
        c1 = curve[min(curve)]
    return curve, c1


def to_pil(x):
    from PIL import Image
    a = x.detach().float().clamp(-1, 1)
    while a.dim() > 2:
        a = a[0]
    return Image.fromarray(((a + 1) * 127.5).byte().cpu().numpy())


def diff_panel(a, b, gain):
    """Absolute difference, dark on white, at a gain shared across columns."""
    d = (a - b).abs() / 2.0
    return 1 - 2 * (gain * d).clamp(0, 1)  # back to [-1, 1] for to_pil


def lpips_pairs(lp, a, b, device, batch=8):
    """LPIPS over aligned batches of 1-channel [-1,1] renders."""
    out = []
    for i in range(0, len(a), batch):
        x = a[i:i + batch].to(device).repeat(1, 3, 1, 1)
        y = b[i:i + batch].to(device).repeat(1, 3, 1, 1)
        out.append(lp(x, y).flatten().cpu())
    return torch.cat(out)


def stat(v):
    """Mean and standard error of a 1-D tensor."""
    v = v.float()
    return float(v.mean()), float(v.std(unbiased=True) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generator-config', type=Path,
                    default=REPO / 'campaign_unsb_no_noise_1.toml',
                    help='the generator whose distance scale is being shown')
    ap.add_argument('--reference-config', type=Path,
                    default=REPO / 'campaign_unsb_full_1.toml',
                    help='stochastic model supplying the trajectory-noise floor, '
                         'the one distance in the figure with a known meaning')
    ap.add_argument('--coverage-csv', type=Path, default=None,
                    help="the run's coverage CSV; its C(N) become the targets")
    ap.add_argument('--levels', type=int, nargs='+', default=LEVELS,
                    help='pool sizes whose C(N) are shown, coarsest first')
    ap.add_argument('--print', dest='print_name', default=None,
                    help="file stem of the shoeprint; default: the chapter's "
                         'worked example, as the other qualitative figures use')
    ap.add_argument('--m', type=int, default=M)
    ap.add_argument('--k', type=int, default=K,
                    help='pairs averaged per level; one pair per level cannot '
                         'distinguish a real ordering from sampling noise')
    ap.add_argument('--batch', type=int, default=16,
                    help='keep <=16 on ROCm, where MIOpen falls off its fast path')
    ap.add_argument('--diff-gain', type=float, default=DIFF_GAIN)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    handler = build_handler('unsb', device, args.generator_config)

    from torchvision.models.inception import inception_v3
    inception = inception_v3(weights='DEFAULT').to(device).eval()
    inception.fc = torch.nn.Identity()
    import lpips
    lp = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    prints = sorted(ss.PRINT_DIR.rglob('*.png')) + sorted(ss.PRINT_DIR.rglob('*.jpg'))
    path = ss.example_print(prints, args.print_name)
    print_img = load_print(path)
    print('shoeprint: %s' % path.name, flush=True)

    g = torch.Generator().manual_seed(0)
    zs = torch.randn(args.m, handler.opt.style_dim, generator=g)
    with torch.no_grad():
        imgs = render(handler, device, print_img, args.m, zs, batch=args.batch)
        feats = embed(imgs, inception, device, batch=args.batch)
    dist = 1 - feats @ feats.T

    # the yardstick: K styles each rendered twice under the stochastic model,
    # so every pair differs by trajectory noise and nothing else
    ref = build_handler('unsb', device, args.reference_config)
    g_ref = torch.Generator().manual_seed(0)
    z_ref = torch.randn(args.k, ref.opt.style_dim,
                        generator=g_ref).repeat_interleave(2, dim=0)
    with torch.no_grad():
        ref_imgs = render(ref, device, print_img, 2 * args.k, z_ref, batch=args.batch)
        f_ref = embed(ref_imgs, inception, device, batch=args.batch)
        ref_cos = 1 - (f_ref[0::2] * f_ref[1::2]).sum(1)
        ref_lp = lpips_pairs(lp, ref_imgs[0::2], ref_imgs[1::2], device)
    floor = float(ref_cos.mean())

    targets = []
    if args.coverage_csv:
        curve, c1 = read_coverage(args.coverage_csv)
        for n in args.levels:
            if n in curve:
                targets.append(('C(%d)' % n, curve[n]))
            else:
                print('skipping level %d: not in %s' % (n, args.coverage_csv.name))
        targets.append(('5% of C(1)\n(the N_95 threshold)', 0.05 * c1))
    else:
        off_all = dist[torch.triu(torch.ones_like(dist), diagonal=1) > 0]
        targets = [('%.3f' % v, float(v)) for v in
                   torch.quantile(off_all, torch.tensor([0.99, 0.5, 0.1, 0.02, 0.005]))]
    targets.append((REF_LABEL, floor))
    targets.sort(key=lambda t: -t[1])

    iu = torch.triu_indices(args.m, args.m, offset=1)
    off = dist[iu[0], iu[1]]
    g_pairs = torch.Generator().manual_seed(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    to_pil(print_img * 2 - 1).save(args.out_dir / 'shoeprint.png')

    cols, rows = [], []
    for k, (label, target) in enumerate(targets):
        if label == REF_LABEL:
            cos_v, lp_v = ref_cos, ref_lp
            pick = int((ref_cos - floor).abs().argmin())
            a, b = ref_imgs[2 * pick], ref_imgs[2 * pick + 1]
        else:
            # sample K from the band nearest the target rather than the K
            # nearest, which would keep reusing the same few styles
            order = (off - target).abs().argsort()
            band = order[:min(len(order), 8 * args.k)]
            sel = band[torch.randperm(len(band), generator=g_pairs)[:args.k]]
            i, j = iu[0][sel], iu[1][sel]
            cos_v = off[sel]
            with torch.no_grad():
                lp_v = lpips_pairs(lp, imgs[i], imgs[j], device)
            a, b = imgs[iu[0][order[0]]], imgs[iu[1][order[0]]]
        cos_m, cos_e = stat(cos_v)
        lp_m, lp_e = stat(lp_v)
        d = diff_panel(a, b, args.diff_gain)
        stem = 'level_%d' % k
        to_pil(a[None]).save(args.out_dir / ('%s_a.png' % stem))
        to_pil(b[None]).save(args.out_dir / ('%s_b.png' % stem))
        to_pil(d[None]).save(args.out_dir / ('%s_diff.png' % stem))
        cols.append((label, a, b, d, label == REF_LABEL))
        rows.append((label.replace('\n', ' '), target, cos_m, cos_e, lp_m, lp_e))

    print('\n%-34s %8s %17s %17s' % ('level', 'target', 'cosine', 'LPIPS'))
    for label, target, cos_m, cos_e, lp_m, lp_e in rows:
        print('%-34s %8.4f  %.4f +/- %.4f  %.4f +/- %.4f'
              % (label, target, cos_m, cos_e, lp_m, lp_e))
    with (args.out_dir / 'levels.csv').open('w') as fh:
        fh.write('level,target,cosine_mean,cosine_se,lpips_mean,lpips_se\n')
        for r in rows:
            fh.write('%s,%.6f,%.6f,%.6f,%.6f,%.6f\n' % r)
        fh.write('# k,%d\n# m,%d\n# print,%s\n' % (args.k, args.m, path.name))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(cols), figsize=(1.95 * len(cols), 10.8))
    axes = np.atleast_2d(axes)
    for c, ((label, a, b, d, is_ref), r) in enumerate(zip(cols, rows)):
        colour = 'tab:red' if is_ref else 'black'
        for row, img in enumerate((a, b, d)):
            ax = axes[row, c]
            ax.imshow(np.asarray(to_pil(img[None])), cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(2.0 if is_ref else 0.8)
        axes[0, c].set_title('%s\n%.4f$\\pm$%.4f\nLPIPS %.3f$\\pm$%.3f'
                             % (label, r[2], r[3], r[4], r[5]),
                             fontsize=8, color=colour)
    axes[0, 0].set_ylabel('render A', fontsize=8)
    axes[1, 0].set_ylabel('render B', fontsize=8)
    axes[2, 0].set_ylabel(r'$|A-B| \times %g$' % args.diff_gain, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'scale.png', dpi=150)
    print('\nwrote %s' % args.out_dir)


if __name__ == '__main__':
    main()
