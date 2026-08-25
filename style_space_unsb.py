"""Style-space analysis for the style-conditioned UNSB.

Ports the analysis of the one_to_many_gan (src/analyse_style_space.py and
src/style_space_rigour.py) to the bridge, with two deliberate differences:

  * sparsity is replaced by a sign-structure analysis. The GAN's mapping
    network ends in a ReLU, so its style space is a non-negative cone by
    construction and counting exact zeros is meaningful; the UNSB's ends in an
    unconstrained linear layer and has no exact zeros. The question here is
    whether it concentrates into a cone anyway, without being forced to.
  * every render shares one trajectory-noise seed, so LPIPS distances between
    renders measure the effect of the style alone rather than the style plus
    the bridge's stochasticity.

The theta-interaction analysis does not port: there is no theta in the bridge.

    style_space_unsb.py [--runs ...] [--out-dir ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from models.ncsn_networks import Conv2dWeightModulate  # noqa: E402
from models.networks import define_SE  # noqa: E402
from unsb_handler import GeneratorHandler, load_unsb_opt  # noqa: E402
from util.evaluation import sb_translate  # noqa: E402

PRINT_DIR = Path("~/Vault/University/Doctorate/Data/Siamese/Compiled/Shoeprints/train").expanduser()
# the adopted model is the deterministic bridge, so that is the default here;
# the stochastic ablations must be analysed with an explicit --generator-config
DEFAULT_CONFIG = REPO / "config_no_noise.toml"
N_STRUCT = 4000     # styles for PCA and sign structure
N_RENDER = 600      # styles rendered for output-based analysis
LPIPS_SIZE = (128, 64)
TRAJ_SEED = 1234    # shared by every render, so style is the only difference
K_NN = 10
N_PAIRS = 4000


# --------------------------------------------------------------------------- #
# Model and sampling
# --------------------------------------------------------------------------- #


def set_key(text, section, key, value):
    out, in_sec, done = [], False, False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_sec = s.startswith("[%s]" % section)
        elif in_sec and not done and s.split("=")[0].strip() == key:
            line, done = "%s = %s" % (key, value), True
        out.append(line)
    if not done:
        raise RuntimeError("key %s not found in [%s]" % (key, section))
    return "\n".join(out) + "\n"


def best_fid_epoch(run):
    import csv
    rows = [r for r in csv.DictReader((REPO / "checkpoints" / run / "metrics.csv").open()) if r["fid"]]
    return min(rows, key=lambda r: float(r["fid"]))["epoch"]


def build(run, device, config=None):
    """Load a run at its best-FID checkpoint.

    config selects the base options the run was trained under. bridge_noise must
    match between training and inference or the analysis renders a model that was
    never trained, so the stochastic configurations of the ablation need an
    explicit config.toml; the default is the adopted deterministic bridge.
    """
    epoch = best_fid_epoch(run)
    text = set_key((config or DEFAULT_CONFIG).read_text(), "experiment", "name", '"%s"' % run)
    text = set_key(text, "test", "epoch", '"%s"' % epoch)
    patched = REPO / ("style_space_%s.toml" % run)
    patched.write_text(text)
    opt = load_unsb_opt(patched)
    handler = GeneratorHandler(opt, device)
    netSE = define_SE(opt.output_nc, opt.style_dim, [], opt).to(device)
    netSE.load_state_dict(torch.load(
        Path(opt.checkpoints_dir) / opt.name / ("%s_net_SE.pth" % opt.epoch),
        map_location=device, weights_only=True))
    netSE.eval()
    patched.unlink()
    return handler, netSE, opt, epoch


def load_image(path, size=(512, 256)):
    x = torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0)[None]
    if x.shape[-2:] != size:
        x = F.interpolate(x[None], size, mode="bicubic", align_corners=False, antialias=True)[0]
    return x.clamp(0, 1)


def example_print(paths, name=None, seed=0):
    """The shoeprint used for the chapter's worked example.

    Chapter figures that follow one shoeprint through the model share this
    choice, so the architecture diagram and the NFE trajectory show the same
    translation. name selects by file stem; without one the seeded shuffle is
    used, which is how the other qualitative figures pick their input.
    """
    if name is not None:
        hit = [p for p in paths if p.stem == name]
        if not hit:
            raise SystemExit("no shoeprint named %s in %s" % (name, PRINT_DIR))
        return hit[0]
    g = torch.Generator().manual_seed(seed)
    return paths[torch.randperm(len(paths), generator=g)[0]]


@torch.no_grad()
def sample_styles(handler, n, device, seed=0):
    """z ~ N(0, I) mapped to the learned style space (thesis: w -> s)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, handler.opt.style_dim, generator=g).to(device)
    out = [handler.generator.z_transform(z[i:i + 512]) for i in range(0, n, 512)]
    return z.cpu().numpy(), torch.cat(out).cpu().numpy()


@torch.no_grad()
def render_fixed_noise(handler, print_img, styles, device):
    """One render per style, every one drawing the same trajectory noise.

    Rendered one at a time: sb_translate draws randn_like over the whole batch,
    so batching would give each item its own trajectory and reintroduce exactly
    the noise this is meant to hold constant.
    """
    src = print_img[None].to(device)
    outs = []
    for i in tqdm(range(len(styles)), desc="render", unit="style", leave=False):
        w = torch.as_tensor(styles[i: i + 1], dtype=torch.float32, device=device)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(TRAJ_SEED)
            outs.append(sb_translate(handler.generator, src, handler.opt,
                                     style=w, style_is_mapped=True).cpu())
    return torch.cat(outs)


# --------------------------------------------------------------------------- #
# 1. PCA structure   2. sign structure
# --------------------------------------------------------------------------- #


def pca_structure(s):
    from sklearn.decomposition import PCA
    return np.cumsum(PCA().fit(s).explained_variance_ratio_).tolist()


def sign_structure(s):
    """Does the space concentrate into a cone without a ReLU forcing it to?

    Reported: the per-component probability of a positive value (0.5 = balanced);
    the occupancy of the most populated sign orthant against the 2^-d expected
    if components were independent and balanced; and the alignment of styles
    with their own mean direction, which is 1 for a narrow cone and ~0 for a
    sign-symmetric cloud.
    """
    d = s.shape[1]
    p_pos = (s > 0).mean(axis=0)
    codes = ((s > 0) * (1 << np.arange(d))).sum(axis=1)
    counts = np.bincount(codes, minlength=1 << d)
    unit = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
    mean_dir = unit.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir) + 1e-12
    cos = unit @ mean_dir
    return {
        "p_positive_per_component": p_pos.tolist(),
        "p_positive_min": float(p_pos.min()),
        "p_positive_max": float(p_pos.max()),
        "max_orthant_occupancy": float(counts.max() / len(s)),
        "orthants_occupied": int((counts > 0).sum()),
        "orthants_possible": int(1 << d),
        "uniform_orthant_occupancy": float(1.0 / (1 << d)),
        "mean_cosine_to_mean_direction": float(cos.mean()),
        "frac_positive_cosine": float((cos > 0).mean()),
        "participation_ratio": float(
            ((s ** 2).sum(axis=1) ** 2 / ((s ** 4).sum(axis=1) + 1e-24)).mean()),
        "energy_top1": float(np.sort(s ** 2, axis=1)[:, ::-1][:, :1].sum(axis=1).mean()
                             / (s ** 2).sum(axis=1).mean()),
        "energy_top2": float(np.sort(s ** 2, axis=1)[:, ::-1][:, :2].sum(axis=1).mean()
                             / (s ** 2).sum(axis=1).mean()),
        "energy_top3": float(np.sort(s ** 2, axis=1)[:, ::-1][:, :3].sum(axis=1).mean()
                             / (s ** 2).sum(axis=1).mean()),
        "frac_exact_zeros": float((s == 0).mean()),
    }


# --------------------------------------------------------------------------- #
# 3. Smoothness
# --------------------------------------------------------------------------- #


@torch.no_grad()
def lpips_pairs(lp, imgs, pairs, device, batch=256):
    out = []
    for start in range(0, len(pairs), batch):
        p = pairs[start:start + batch]
        a = imgs[p[:, 0]].to(device).repeat(1, 3, 1, 1)
        b = imgs[p[:, 1]].to(device).repeat(1, 3, 1, 1)
        out.append(lp(a, b).flatten().cpu().numpy())
    return np.concatenate(out)


def morans_i(values, coords, k, rng, n_perm=999):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]

    def stat(v):
        z = v - v.mean()
        return float((z * z[idx].sum(axis=1)).sum() / (k * (z ** 2).sum()))

    obs = stat(values)
    null = np.array([stat(rng.permutation(values)) for _ in range(n_perm)])
    return obs, float((1 + (null >= obs).sum()) / (1 + n_perm))


def knn_pairs(coords, k):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    return np.stack([np.repeat(np.arange(len(coords)), k), idx.reshape(-1)], axis=1)


def random_pairs(n, count, rng):
    p = rng.integers(0, n, size=(int(count * 1.3), 2))
    return p[p[:, 0] != p[:, 1]][:count]


def smoothness(lp, small, coords_by_space, coverage, contrast, device, rng):
    from scipy.stats import spearmanr
    n = len(small)
    rand_p = random_pairs(n, n * K_NN, rng)
    lp_rand = lpips_pairs(lp, small, rand_p, device).mean()
    corr_p = random_pairs(n, N_PAIRS, rng)
    lp_corr = lpips_pairs(lp, small, corr_p, device)

    out = {}
    for name, coords in coords_by_space.items():
        nbr = knn_pairs(coords, K_NN)
        lp_nbr = lpips_pairs(lp, small, nbr, device).mean()
        cd = np.linalg.norm(coords[corr_p[:, 0]] - coords[corr_p[:, 1]], axis=1)
        mi_cov, p_cov = morans_i(coverage, coords, K_NN, rng)
        mi_con, p_con = morans_i(contrast, coords, K_NN, rng)
        out[name] = {
            "morans_intensity": mi_cov, "morans_intensity_p": p_cov,
            "morans_contrast": mi_con, "morans_contrast_p": p_con,
            "spearman_lpips": float(spearmanr(cd, lp_corr).statistic),
            "knn_coherence": float(lp_rand / lp_nbr),
        }
    return out


# --------------------------------------------------------------------------- #
# 4-6. Figures
# --------------------------------------------------------------------------- #


def tsne_figures(s, coverage, contrast, out_dir, tag):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perp = min(30, max(5, len(s) // 4))
    emb = TSNE(n_components=2, init="pca", perplexity=perp, random_state=0).fit_transform(s)
    for name, vals, label in (("norm", np.linalg.norm(s, axis=1), "style norm"),
                              ("coverage", coverage, "mean output intensity"),
                              ("contrast", contrast, "output contrast")):
        plt.figure(figsize=(4, 4))
        sc = plt.scatter(emb[:, 0], emb[:, 1], c=vals, s=6, cmap="viridis")
        plt.colorbar(sc, label=label)
        plt.xticks([]); plt.yticks([])
        for sp in plt.gca().spines.values():
            sp.set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / ("s_t_sne_%s_%s.png" % (name, tag)), dpi=200)
        plt.close()


def sefa_directions(handler, k):
    mats = [m.to_style.weight().detach().cpu().numpy()
            for m in handler.generator.modules() if isinstance(m, Conv2dWeightModulate)]
    a = np.concatenate(mats, axis=0)
    eigvals, eigvecs = np.linalg.eigh(a.T @ a)
    order = eigvals.argsort()[::-1]
    return eigvecs[:, order[:k]].T, eigvals[order[:k]]


def sefa_walk(handler, print_img, s, out_dir, tag, device, k=4, steps=7):
    import torchvision
    dirs, _ = sefa_directions(handler, k)
    base, sd = s.mean(axis=0), s.std(axis=0)
    tiles = []
    for d in dirs:
        scale = float(np.linalg.norm(sd * d))
        for t in np.linspace(-2 * scale, 2 * scale, steps):
            tiles.append(render_fixed_noise(handler, print_img, (base + t * d)[None], device))
    grid = torchvision.utils.make_grid(torch.cat(tiles), nrow=steps, padding=2, normalize=True)
    torchvision.utils.save_image(grid, out_dir / ("walk_sefa_%s.png" % tag))


def real_overlay(netSE, s, val_dir, out_dir, tag, device):
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(val_dir.rglob("*.png")) + sorted(val_dir.rglob("*.jpg"))
    if not files:
        return {"n_real": 0}
    with torch.no_grad():
        real = torch.cat([netSE(load_image(f)[None].to(device) * 2 - 1).cpu() for f in files]).numpy()
    pca = PCA(n_components=2).fit(s)
    a, b = pca.transform(s), pca.transform(real)
    plt.figure(figsize=(5, 4))
    plt.scatter(a[:, 0], a[:, 1], s=6, c="0.7", label="sampled")
    plt.scatter(b[:, 0], b[:, 1], s=10, c="tab:red", label="extracted from held-out marks")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / ("real_overlay_pca_%s.png" % tag), dpi=200)
    plt.close()
    lo, hi = a.min(axis=0), a.max(axis=0)
    return {"n_real": len(files),
            # the projection the overlay is judged in, so how much of the space
            # it actually shows belongs with the containment fraction
            "pc12_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "frac_real_inside_sampled_pc_range": float(((b >= lo) & (b <= hi)).all(axis=1).mean()),
            "real_mean_norm": float(np.linalg.norm(real, axis=1).mean()),
            "sampled_mean_norm": float(np.linalg.norm(s, axis=1).mean())}


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["shoeprint_unsb_v2_no_noise_1", "shoeprint_unsb_v2_no_noise_2",
                             "shoeprint_unsb_v2_no_noise_3"])
    ap.add_argument("--out-dir", type=Path, default=REPO / "style_analysis")
    ap.add_argument("--figures-for", default="shoeprint_unsb_v2_no_noise_1")
    ap.add_argument("--generator-config", type=Path, default=None,
                    help="base config the runs were trained under (default: %s); pass "
                         "config.toml for the stochastic-bridge ablations" % DEFAULT_CONFIG.name)
    ap.add_argument("--out-json", type=Path, default=None,
                    help="results file (default: <out-dir>/style_space_unsb.json); the "
                         "figures carry their run tag but this file does not, so a second "
                         "configuration needs its own name to avoid overwriting the first")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    files = sorted(PRINT_DIR.rglob("*.png")) + sorted(PRINT_DIR.rglob("*.jpg"))
    g = torch.Generator().manual_seed(0)
    print_img = load_image(files[torch.randperm(len(files), generator=g)[0]])

    results = {}
    for run in args.runs:
        handler, netSE, opt, epoch = build(run, device, args.generator_config)
        tag = run.replace("shoeprint_unsb_v2", "unsb")
        print("\n=== %s @ epoch %s, bridge_noise %s ==="
              % (run, epoch, getattr(opt, "bridge_noise", 1.0)), flush=True)

        z_big, s_big = sample_styles(handler, N_STRUCT, device, seed=0)
        z, s = z_big[:N_RENDER], s_big[:N_RENDER]

        imgs = render_fixed_noise(handler, print_img, s, device)
        coverage = imgs.mean(dim=(1, 2, 3)).numpy()
        contrast = imgs.std(dim=(1, 2, 3)).numpy()
        small = F.interpolate(imgs, LPIPS_SIZE, mode="bilinear", align_corners=False)

        rng = np.random.default_rng(0)
        from scipy.stats import spearmanr
        res = {
            "epoch": epoch,
            "pca_cumulative_variance": pca_structure(s_big),
            "sign_structure": sign_structure(s_big),
            # does style magnitude carry severity, as theta does by construction
            # in the GAN? Nothing in the bridge encourages it, so it is measured
            # rather than assumed; this is the statistic behind the norm panel
            "norm_intensity_spearman": float(
                spearmanr(np.linalg.norm(s, axis=1), coverage).statistic),
            "smoothness": smoothness(lp, small, {"s": s, "w": z}, coverage, contrast, device, rng),
        }
        if run == args.figures_for:
            tsne_figures(s, coverage, contrast, args.out_dir, tag)
            sefa_walk(handler, print_img, s_big, args.out_dir, tag, device)
            res["real_overlay"] = real_overlay(
                netSE, s_big, Path(opt.dir_B).expanduser() / "val", args.out_dir, tag, device)
        results[run] = res
        print(json.dumps(res, indent=2)[:1200], flush=True)
        del handler, netSE
        torch.cuda.empty_cache()

    out_json = args.out_json or args.out_dir / "style_space_unsb.json"
    out_json.write_text(json.dumps(results, indent=2))
    print("\nwrote %s" % out_json)


if __name__ == "__main__":
    main()
