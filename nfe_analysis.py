"""Translation quality and character as a function of the number of function
evaluations, for the style-conditioned UNSB.

FID/KID at every NFE in the sweep, CIS at a subset (it costs a second 10,000
images), and a qualitative strip of one shoeprint under one style rendered at
each NFE with the trajectory seed held fixed.

The model is trained with five timesteps, so NFE above five extrapolates: the
timestep conditioning is a sinusoidal embedding and remains well defined, but
the generator has never seen those indices. This is the regime in which Kim
et al. report over-translation.

    nfe_analysis.py [--run ...] [--nfe 1 2 3 4 5 6 8 10] [--cis-at 1 3 5 10]
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision

import style_space_unsb as ss
from util import evaluation
from util.evaluation import sb_translate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shoeprint_unsb_v2_no_noise_1")
    ap.add_argument("--generator-config", type=Path, default=None,
                    help="base config the run was trained under (default: %s); pass "
                         "config.toml for the stochastic-bridge ablations"
                         % ss.DEFAULT_CONFIG.name)
    ap.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--cis-at", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--example-print", default=None,
                    help="file stem of the shoeprint for the strip; default: the seeded choice")
    ap.add_argument("--example-style-seed", type=int, default=0)
    # torch.randn is not prefix stable, so the size of the draw is part of the
    # choice: the same index of a different count is a different style
    ap.add_argument("--example-style-count", type=int, default=6)
    ap.add_argument("--example-style-index", type=int, default=0)
    ap.add_argument("--strip-only", action="store_true",
                    help="write the qualitative strip and skip the metric sweep")
    ap.add_argument("--no-strip", action="store_true",
                    help="skip the qualitative strip; use when topping up metrics so a "
                         "published figure is not silently redrawn with new defaults")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute metrics already present in the output file "
                         "(the default tops up only what is missing)")
    args = ap.parse_args()
    if args.out is None:
        args.out = ss.REPO / "style_analysis" / ("nfe_analysis_%s.json" % args.run)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    handler, netSE, opt, epoch = ss.build(args.run, device, args.generator_config)
    netG = handler.generator
    trained_T = opt.num_timesteps
    print("%s @ epoch %s, trained with %d timesteps, bridge_noise %s"
          % (args.run, epoch, trained_T, getattr(opt, "bridge_noise", 1.0)), flush=True)

    # qualitative strip first: cheap, and useful even if the sweep is interrupted
    if not args.no_strip:
        prints = sorted(ss.PRINT_DIR.rglob("*.png")) + sorted(ss.PRINT_DIR.rglob("*.jpg"))
        img = ss.load_image(ss.example_print(prints, args.example_print))
        k = args.example_style_index
        _, styles = ss.sample_styles(handler, args.example_style_count, device,
                                     seed=args.example_style_seed)
        w = torch.as_tensor(styles[k:k + 1], dtype=torch.float32, device=device)
        tiles = [(img * 2 - 1)[None]]
        for n in args.nfe:
            opt.num_timesteps = n
            with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(ss.TRAJ_SEED)
                tiles.append(sb_translate(netG, img[None].to(device), opt,
                                          style=w, style_is_mapped=True).cpu())
        grid = torchvision.utils.make_grid(torch.cat(tiles), nrow=len(tiles), padding=2, normalize=True)
        strip = ss.REPO / "style_analysis" / ("nfe_sweep_%s.png" % args.run)
        torchvision.utils.save_image(grid, strip)
        print("wrote %s" % strip, flush=True)
    if args.strip_only:
        return

    # top up rather than recompute: a sweep point costs 10,000 translations for
    # FID/KID and another 10,000 for CIS, so adding a column to the table must
    # not repay the cost of the columns already in it
    results = {"run": args.run, "epoch": epoch, "trained_timesteps": trained_T, "nfe": {}}
    if args.out.exists() and not args.recompute:
        prior = json.loads(args.out.read_text())
        if str(prior.get("epoch")) != str(epoch):
            raise SystemExit(
                "%s holds epoch %s but this run loads epoch %s; topping up would mix "
                "checkpoints in one table. Pass --recompute or a fresh --out."
                % (args.out, prior.get("epoch"), epoch))
        results["nfe"] = prior.get("nfe", {})
    for n in args.nfe:
        opt.num_timesteps = n
        row = results["nfe"].setdefault(str(n), {})
        if "fid" not in row or "kid" not in row:
            row["fid"], row["kid"] = evaluation.validate_kid_fid(netG, opt, device)
        if n in args.cis_at and "cis" not in row:
            row["cis"] = evaluation.validate_cis(netG, opt, device)
        print("NFE %2d  FID %7.3f  KID %.5f  CIS %s"
              % (n, row["fid"], row["kid"], ("%.4f" % row["cis"]) if "cis" in row else "--"), flush=True)
        args.out.write_text(json.dumps(results, indent=2))  # checkpoint after each

    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
