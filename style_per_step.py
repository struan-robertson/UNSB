"""Is the style already fully rendered at the first bridge step?

The style cycle loss is applied to the endpoint prediction made at one
uniformly sampled timestep (sb_model.py:205, 385-388), and the style extractor
is not timestep-conditioned, so it holds the same standard at every step. This
asks what that produces: for each step of the trajectory, how well the injected
style can be recovered from that step's prediction, and how much of the
variation between renders of one shoeprint that step's style accounts for
relative to the trajectory noise.

Style saturating at step 0 means the extra function evaluations refine the mark
but not its style; style improving along the trajectory means the current
design already defers it.

Renders are one at a time with the trajectory noise seeded per trajectory, so
the same noise is drawn for every style (sb_translate draws randn_like over the
whole batch, which would otherwise give each item its own trajectory). The
first step carries no trajectory noise by construction, so its trajectory share
is exactly zero.

    uv run python style_per_step.py [--runs ...] [--prints 6] [--styles 8]
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

import style_space_unsb as ss
from util.evaluation import sb_translate


def decompose(x):
    """Shares of the total pixel variance explained by style and by trajectory.

    x is [style, traj, pixels] for one shoeprint at one step. Held at full
    resolution: downsampling averages away the high-frequency trajectory
    speckle while preserving style's global tone, which biases the split.
    The remainder is the style-trajectory interaction.
    """
    total = x.reshape(-1, x.shape[-1]).var(0, unbiased=False).mean()
    by_style = x.mean(1).var(0, unbiased=False).mean()
    by_traj = x.mean(0).var(0, unbiased=False).mean()
    return (by_style / total).item(), (by_traj / total).item()


@torch.no_grad()
def measure(run, device, n_prints, n_styles, n_trajs, config=None):
    handler, netSE, opt, epoch = ss.build(run, device, config)
    netG = handler.generator
    T = opt.num_timesteps
    print("%s @ epoch %s" % (run, epoch), flush=True)

    paths = sorted(ss.PRINT_DIR.rglob("*.png")) + sorted(ss.PRINT_DIR.rglob("*.jpg"))
    g = torch.Generator().manual_seed(0)
    paths = [paths[i] for i in torch.randperm(len(paths), generator=g)[:n_prints]]
    _, styles = ss.sample_styles(handler, n_styles, device, seed=0)
    W = torch.as_tensor(styles, dtype=torch.float32, device=device)

    recovered = [[] for _ in range(T)]      # extracted styles, all prints
    shares = [[] for _ in range(T)]         # (style, traj) variance shares per print
    intensity = [[] for _ in range(T)]

    for p_i, path in enumerate(paths):
        src = (ss.load_image(path) * 2 - 1)[None].to(device)
        # one print's renders at a time: full-resolution pixels for the variance
        # decomposition are ~13 MB per step and are reduced before the next print
        images = [[] for _ in range(T)]
        for k in range(n_styles):
            for m in range(n_trajs):
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(ss.TRAJ_SEED + m)
                    steps = sb_translate(netG, src, opt, style=W[k:k + 1],
                                         style_is_mapped=True, return_steps=True)
                for t, x in enumerate(steps):
                    recovered[t].append(netSE(x)[0].float().cpu())
                    images[t].append(x[0, 0].float().flatten().cpu())
        for t in range(T):
            x = torch.stack(images[t]).reshape(n_styles, n_trajs, -1)
            shares[t].append(decompose(x))
            intensity[t].append((x.mean() * 0.5 + 0.5).item())
        print("  print %d/%d" % (p_i + 1, len(paths)), flush=True)

    # style k was rendered for n_trajs trajectories, for each print in turn
    target = W.cpu().repeat_interleave(n_trajs, dim=0).repeat(len(paths), 1)
    truth = torch.arange(n_styles).repeat_interleave(n_trajs).repeat(len(paths))
    rows = []
    for t in range(T):
        s_hat = torch.stack(recovered[t])
        cos_err = (1 - F.cosine_similarity(F.normalize(target, dim=-1),
                                           F.normalize(s_hat, dim=-1), dim=-1)).mean()
        l2 = F.mse_loss(target, s_hat)

        # which injected style does the recovered one look most like?
        sim = F.normalize(s_hat, dim=-1) @ F.normalize(W.cpu(), dim=-1).T
        top1 = (sim.argmax(dim=1) == truth).float().mean()

        v_style = sum(s for s, _ in shares[t]) / len(shares[t])
        v_traj = sum(v for _, v in shares[t]) / len(shares[t])
        rows.append(dict(run=run, epoch=epoch, step=t + 1,
                         cos_err="%.4f" % cos_err, l2="%.4f" % l2,
                         cycle_loss="%.4f" % (cos_err + opt.style_cos_l2_ratio * l2),
                         top1="%.4f" % top1,
                         var_share_style="%.4f" % v_style,
                         var_share_traj="%.4f" % v_traj,
                         var_share_interaction="%.4f" % (1 - v_style - v_traj),
                         mean_intensity="%.4f" % (sum(intensity[t]) / len(intensity[t]))))
        print("  step %d: cycle %.4f  top1 %.3f  var style %.3f traj %.3f inter %.3f"
              % (t + 1, cos_err + opt.style_cos_l2_ratio * l2, top1,
                 v_style, v_traj, 1 - v_style - v_traj), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["shoeprint_unsb_v2_no_noise_1", "shoeprint_unsb_v2_no_noise_2",
                             "shoeprint_unsb_v2_no_noise_3"])
    ap.add_argument("--generator-config", type=Path, default=None,
                    help="base config the runs were trained under (default: %s); pass "
                         "config.toml for the stochastic-bridge runs" % ss.DEFAULT_CONFIG.name)
    ap.add_argument("--prints", type=int, default=6)
    ap.add_argument("--styles", type=int, default=8)
    ap.add_argument("--trajs", type=int, default=3)
    # not --out: the UNSB options parser also reads sys.argv, where --out
    # prefix-matches its --output_nc
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    out = args.csv or (ss.REPO / "style_analysis" / "style_per_step.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for run in args.runs:
        rows += measure(run, device, args.prints, args.styles, args.trajs,
                        args.generator_config)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
