"""Qualitative figures for the UNSB chapter, mirroring those of gan.tex.

Styles shown are chosen by farthest-point (maximin) selection over LPIPS
distance between their rendered outputs, so the marks displayed are as visually
distinct as the sampled style space allows rather than an arbitrary sample.
Every render in a style-comparison figure shares one trajectory-noise seed, so
what the reader sees is attributable to the style and not to the bridge's
stochasticity; the NFE figure is the exception, being about the trajectory.

    qualitative_unsb.py [--run ...] [--n-styles 4] [--out-dir ...]
"""

import argparse
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import style_space_unsb as ss
from util.evaluation import sb_translate

N_CAND = 256          # candidate styles the selection draws from
LPIPS_SIZE = (128, 64)


def to_pil(x):
    a = x.detach().float().clamp(-1, 1)
    while a.dim() > 2:  # accept [H,W], [C,H,W] and [B,C,H,W]
        a = a[0]
    return Image.fromarray(((a + 1) * 127.5).byte().cpu().numpy())


def save(x, path):
    to_pil(x).save(path)


@torch.no_grad()
def maximin_styles(handler, print_img, styles, n, device, lp):
    """Greedy farthest-point selection over LPIPS distance between renders."""
    imgs = ss.render_fixed_noise(handler, print_img, styles, device)
    small = F.interpolate(imgs, LPIPS_SIZE, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)

    def dist_to(i):
        out = []
        for s in range(0, len(small), 128):
            b = small[s:s + 128].to(device)
            out.append(lp(small[i][None].to(device).expand(len(b), -1, -1, -1), b).flatten().cpu().numpy())
        return np.concatenate(out)

    chosen = [0]
    mind = dist_to(0)
    while len(chosen) < n:
        nxt = int(mind.argmax())
        chosen.append(nxt)
        mind = np.minimum(mind, dist_to(nxt))
    return styles[chosen], imgs[chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shoeprint_unsb_v2_no_noise_1")
    ap.add_argument("--generator-config", type=Path, default=None,
                    help="base config the run was trained under (default: %s); pass "
                         "config.toml for the stochastic-bridge ablations"
                         % ss.DEFAULT_CONFIG.name)
    ap.add_argument("--n-styles", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=ss.REPO / "style_analysis" / "qualitative")
    ap.add_argument("--example-print", default=None,
                    help="file stem of the shoeprint for the architecture and NFE figures; "
                         "default: the same seeded choice as the other figures")
    ap.add_argument("--example-style-seed", type=int, default=0)
    # torch.randn is not prefix stable, so the size of the draw is part of the
    # choice: the same index of a different count is a different style
    ap.add_argument("--example-style-count", type=int, default=6)
    ap.add_argument("--example-style-index", type=int, default=None,
                    help="index into a seeded style sample; default: the first maximin style")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    handler, netSE, opt, epoch = ss.build(args.run, device, args.generator_config)
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    print("%s @ epoch %s, bridge_noise %s"
          % (args.run, epoch, getattr(opt, "bridge_noise", 1.0)), flush=True)

    prints = sorted(ss.PRINT_DIR.rglob("*.png")) + sorted(ss.PRINT_DIR.rglob("*.jpg"))
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(prints), generator=g)
    main_print = ss.load_image(prints[order[0]])

    for sub in ("styles", "nfe", "extraction", "consistency", "arch"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- (i) one shoeprint, maximally distinct styles -----------------------
    _, cand = ss.sample_styles(handler, N_CAND, device, seed=0)
    sel, sel_imgs = maximin_styles(handler, main_print, cand, args.n_styles, device, lp)
    save(main_print * 2 - 1, args.out_dir / "styles" / "shoeprint.png")
    for i, im in enumerate(sel_imgs, 1):
        save(im[None], args.out_dir / "styles" / ("shoemark_%d.png" % i))

    # --- (ii) NFE refinement trajectory, and the architecture figure's pair --
    ex_print = ss.load_image(ss.example_print(prints, args.example_print))
    if args.example_style_index is None:
        w = torch.as_tensor(sel[:1], dtype=torch.float32, device=device)
    else:
        k = args.example_style_index
        _, drawn = ss.sample_styles(handler, args.example_style_count, device,
                                  seed=args.example_style_seed)
        w = torch.as_tensor(drawn[k:k + 1], dtype=torch.float32, device=device)
    with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(ss.TRAJ_SEED)
        steps = sb_translate(handler.generator, ex_print[None].to(device), opt,
                             style=w, style_is_mapped=True, return_steps=True)
    save(ex_print * 2 - 1, args.out_dir / "nfe" / "shoeprint.png")
    for i, s in enumerate(steps, 1):
        save(s, args.out_dir / "nfe" / ("nfe_%d.png" % i))
    # the architecture diagram shows the same example: input, and the mark the
    # final evaluation predicts from it
    save(ex_print * 2 - 1, args.out_dir / "arch" / "shoeprint.png")
    save(steps[-1], args.out_dir / "arch" / "shoemark.png")

    # --- (iii) example-guided translation from a real shoemark --------------
    val_marks = sorted((Path(opt.dir_B).expanduser() / "val").rglob("*.png"))
    if val_marks:
        donor = ss.load_image(val_marks[torch.randperm(len(val_marks), generator=g)[0]])
        with torch.no_grad():
            w_real = netSE(donor[None].to(device) * 2 - 1)
        save(donor * 2 - 1, args.out_dir / "extraction" / "donor_shoemark.png")
        for k in (1, 2):
            p = ss.load_image(prints[order[k]])
            out = ss.render_fixed_noise(handler, p, w_real.cpu().numpy(), device)
            save(p * 2 - 1, args.out_dir / "extraction" / ("shoeprint_%d.png" % k))
            save(out[:1], args.out_dir / "extraction" / ("shoemark_%d.png" % k))

    # --- (iv) one randomly sampled style across thirty-six shoeprints -------
    _, w_rand = ss.sample_styles(handler, 1, device, seed=1)
    tiles = []
    for k in range(36):
        p = ss.load_image(prints[order[k + 10]])
        tiles.append(ss.render_fixed_noise(handler, p, w_rand[:1], device))
    grid = torchvision.utils.make_grid(torch.cat(tiles), nrow=6, padding=2, normalize=True)
    torchvision.utils.save_image(grid, args.out_dir / "consistency" / "multiple.png")

    # --- (v) the same styles on two shoeprints ------------------------------
    for k in (1, 2):
        p = ss.load_image(prints[order[k]])
        row = ss.render_fixed_noise(handler, p, sel, device)
        grid = torchvision.utils.make_grid(torch.cat([(p * 2 - 1)[None], row]),
                                           nrow=args.n_styles + 1, padding=2, normalize=True)
        torchvision.utils.save_image(grid, args.out_dir / "consistency" / ("single_%d.png" % k))

    print("wrote %s" % args.out_dir)


if __name__ == "__main__":
    main()
