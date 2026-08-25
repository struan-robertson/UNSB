"""KID/FID and CIS validation, mirroring the one_to_many_gan project.

Translations are generated from the domain-A images (each with a freshly
sampled style vector, cycling the set until n_evaluation_images are produced),
saved as PNGs, and scored with cleanfid's compute_fid/compute_kid against the
real domain-B directory. use_training_data selects train/ for both the source
and the reference (the convention of the GAN's definitive test_kid_fid.py
evaluation — FID against the small val/ set is biased sharply upward);
false selects val/ for both (the GAN's train-time hook).
"""
import csv
import gc
import math
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import trange

import util.util as util
from data.twodir_dataset import TwoDirDataset


@contextmanager
def training_state_offloaded(model):
    """Temporarily move everything except the generator weights to the CPU.

    Frees gradients, the non-generator networks, and all Adam moments so that
    validation (generation + cleanfid's InceptionV3) has as much VRAM headroom
    as possible; everything is restored on exit.
    """
    device = model.device
    offload_nets = [getattr(model, 'net' + name) for name in model.model_names if name != 'G']

    for name in model.model_names:
        getattr(model, 'net' + name).zero_grad(set_to_none=True)
    for net in offload_nets:
        net.to('cpu')

    # move only GPU-resident optimizer state (Adam keeps its 'step' counters on
    # the CPU and requires them to stay there), remembering exactly what moved
    moved = []
    for optimizer in model.optimizers:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.device.type != 'cpu':
                    state[key] = value.cpu()
                    moved.append((state, key))

    gc.collect()
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        for net in offload_nets:
            net.to(device)
        for state, key in moved:
            state[key] = state[key].to(device)


def sb_translate(netG, x, opt, style=None, style_is_mapped=False, return_steps=False):
    """Run the full NFE inference loop of the Schrödinger bridge for a batch.

    Mirrors the test-phase loop in sb_model.py: one style vector per sample is
    held fixed across all timesteps. Returns the final translated batch, or the
    output of every timestep when return_steps is set (for the NFE figure).
    """
    device = x.device
    tau = opt.tau
    # decoupled noise multiplier; must match the value the model was trained with
    noise = getattr(opt, 'bridge_noise', 1.0)
    T = opt.num_timesteps
    incs = [0.0] + [1 / (i + 1) for i in range(T - 1)]
    times = torch.tensor(incs, dtype=torch.float32, device=device).cumsum(0)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = torch.cat([torch.zeros(1, device=device), times])

    if style is None:
        style = torch.randn(x.shape[0], opt.style_dim, device=device)

    Xt = x
    x1_hat = x
    steps = []
    autocast = torch.autocast(device.type, dtype=torch.bfloat16,
                              enabled=opt.mixed_precision and device.type == 'cuda')
    with autocast:
        for t in range(T):
            if t > 0:
                delta = times[t] - times[t - 1]
                denom = times[-1] - times[t - 1]
                inter = (delta / denom).reshape(-1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                Xt = (1 - inter) * Xt + inter * x1_hat + noise * (scale * tau).sqrt() * torch.randn_like(Xt)
            time_idx = (t * torch.ones(x.shape[0], device=device)).long()
            x1_hat = netG(Xt, time_idx, style, style_is_mapped=style_is_mapped)
            if return_steps:
                steps.append(x1_hat.float())
    # back to fp32 so callers can stack the result with real (fp32) images
    return steps if return_steps else x1_hat.float()


def _val_dataloader(opt):
    """Domain-A source images for FID/KID, deterministic order, no augmentation.

    use_training_data selects the train/ images (matching the definitive
    test_kid_fid.py evaluation of the one_to_many_gan project, which translates
    all training shoeprints and scores against all training shoemarks);
    otherwise val/ (matching that project's train-time hook).
    """
    phase = 'train' if opt.use_training_data else 'val'
    val_opt = util.copyconf(opt, isTrain=False, phase=phase, serial_batches=True, no_flip=True)
    dataset = TwoDirDataset(val_opt)
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)


def record_metrics(opt, fid_score, kid_score, cis_score='', epoch='', total_iters=''):
    """Append a KID/FID (and optionally CIS) measurement to <checkpoints_dir>/<name>/metrics.csv."""
    path = Path(opt.checkpoints_dir) / opt.name / 'metrics.csv'
    is_new = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['time', 'epoch', 'total_iters', 'fid', 'kid', 'cis'])
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), epoch, total_iters, fid_score, kid_score, cis_score])


def create_image_checkpoints(netG, netSE, opt, device, tag):
    """Save two visual checkpoints under <checkpoints_dir>/<name>/validation/.

    translation_<tag>.png    -- rows are val shoeprints: the input followed by
                                translations with the same fixed random styles
                                at every checkpoint (comparable across training)
    style_transfer_<tag>.png -- rows of [shoeprint, translation in the style
                                extracted from a real shoemark, that shoemark];
                                skipped when netSE is unavailable
    """
    out_dir = Path(opt.checkpoints_dir) / opt.name / 'validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, n_styles = 4, 6

    # fixed random val images for the rows: consecutive files are typically
    # prints/marks of the same shoe, so draw seeded random indices (identical at
    # every checkpoint, so the grids stay comparable) rather than the first rows
    val_opt = util.copyconf(opt, isTrain=False, phase='val', serial_batches=True, no_flip=True)
    dataset = TwoDirDataset(val_opt)
    rows = min(rows, dataset.A_size, dataset.B_size)
    gen = torch.Generator().manual_seed(0)
    idx_A = torch.randperm(dataset.A_size, generator=gen)[:rows]
    idx_B = torch.randperm(dataset.B_size, generator=gen)[:rows]
    A = torch.stack([dataset[i]['A'] for i in idx_A.tolist()]).to(device)
    B = torch.stack([dataset[i]['B'] for i in idx_B.tolist()]).to(device)

    netG.eval()
    with torch.no_grad():
        # 1. one shoeprint per row, translated with the same fixed random styles
        gen = torch.Generator().manual_seed(0)
        zs = torch.randn(n_styles, opt.style_dim, generator=gen).to(device)
        tiles = [A]
        for j in range(n_styles):
            style = zs[j:j + 1].expand(A.shape[0], -1)
            tiles.append(sb_translate(netG, A, opt, style=style))
        grid = torch.stack(tiles, dim=1).flatten(0, 1)
        torchvision.utils.save_image(grid, out_dir / ('translation_%s.png' % tag),
                                     nrow=n_styles + 1, padding=2, normalize=True)

        # 2. shoeprints translated in the style extracted from a real shoemark
        if netSE is not None:
            netSE.eval()
            w = netSE(B)
            transferred = sb_translate(netG, A, opt, style=w, style_is_mapped=True)
            grid = torch.stack([A, transferred, B], dim=1).flatten(0, 1)
            torchvision.utils.save_image(grid, out_dir / ('style_transfer_%s.png' % tag),
                                         nrow=3, padding=2, normalize=True)


def validate_kid_fid(netG, opt, device, frozen_style=False):
    """Generate n_evaluation_images translations and score them with clean-fid.

    Returns (fid_score, kid_score).
    """
    # imported here so the siamese pipeline can import sb_translate without
    # the evaluation-only dependencies (clean-fid, scipy)
    from cleanfid import fid

    generation_dir = Path(opt.checkpoints_dir) / opt.name / 'generated'
    shutil.rmtree(generation_dir, ignore_errors=True)
    generation_dir.mkdir(parents=True, exist_ok=True)

    reference_dir = Path(opt.dir_B).expanduser() / ('train' if opt.use_training_data else 'val')

    dataloader = _val_dataloader(opt)
    data_iter = iter(dataloader)
    # frozen_style: one style per source shoeprint, reused every time the loader
    # cycles, so only the bridge trajectory varies between renders of a print.
    # The loader is unshuffled, so position in the pass identifies the print.
    styles = pos = None
    if frozen_style:
        styles = torch.randn(len(dataloader.dataset), opt.style_dim,
                             generator=torch.Generator().manual_seed(0))
        pos = 0

    netG.eval()
    count = 0
    # seeded, isolated RNG: every evaluation draws identical styles and bridge
    # noise, so scores are comparable across checkpoints (any movement is the
    # model, not sampling); fork_rng keeps the seeding from disturbing the
    # training RNG stream when called from the train-time hook
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(0)
        with torch.no_grad():
            pbar = trange(opt.n_evaluation_images, desc='Generating for FID/KID', unit='img',
                          leave=False, dynamic_ncols=True)
            while count < opt.n_evaluation_images:
                try:
                    data = next(data_iter)
                except StopIteration:  # cycle the val set; each pass draws new styles
                    data_iter = iter(dataloader)
                    data = next(data_iter)
                a = data['A'].to(device)
                z = None
                if frozen_style:
                    z = styles[pos:pos + a.shape[0]].to(device)
                    pos = (pos + a.shape[0]) % len(styles)
                fakes = sb_translate(netG, a, opt, style=z)
                for fake in fakes:
                    if count >= opt.n_evaluation_images:
                        break
                    torchvision.utils.save_image(fake, generation_dir / ('%d.png' % count), normalize=True)
                    count += 1
                    pbar.update(1)
            pbar.close()

    gc.collect()
    torch.cuda.empty_cache()

    # num_workers=0: cleanfid's ResizeDataset holds a closure that cannot be pickled,
    # which crashes multiprocessing workers under Python 3.14's forkserver default
    fid_score = fid.compute_fid(str(generation_dir), str(reference_dir), verbose=False, device=device,
                                use_dataparallel=False, num_workers=0)
    # cleanfid's KID estimator draws its random subsets from numpy's global RNG;
    # seed (and restore) it so the estimate is deterministic like everything else
    np_state = np.random.get_state()
    np.random.seed(0)
    try:
        kid_score = fid.compute_kid(str(generation_dir), str(reference_dir), verbose=False, device=device,
                                    use_dataparallel=False, num_workers=0)
    finally:
        np.random.set_state(np_state)
    return fid_score, kid_score


def validate_cis(netG, opt, device, frozen_style=False):
    """Conditional Inception Score, mirroring the one_to_many_gan's test_cis.py.

    Each of the first (up to) 100 val shoeprints is translated
    cond_is_n_evaluation_images times with per-sample random styles, and the
    Inception Score of that conditional set is computed (splits=1); CIS is the
    mean over shoeprints. Higher means more per-input diversity. Images reach
    InceptionV3 exactly as in the GAN project — bicubic-resized to 299x299,
    replicated to 3 channels, min-max normalised per generation batch, no
    ImageNet normalisation — so scores are comparable between the two projects.
    """
    # lazy for the same reason as the clean-fid import in validate_kid_fid
    from scipy.stats import entropy
    from torchvision.models.inception import inception_v3

    val_opt = util.copyconf(opt, isTrain=False, phase='val', serial_batches=True, no_flip=True)
    dataset = TwoDirDataset(val_opt)
    n_sources = min(100, dataset.A_size)  # the GAN scores one 100-image val batch

    inception = inception_v3(weights='DEFAULT').to(device).eval()

    n_images = opt.cond_is_n_evaluation_images
    batch = opt.batch_size
    netG.eval()
    scores = []
    # seeded and isolated for the same reason as in validate_kid_fid
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []), torch.no_grad():
        torch.manual_seed(0)
        for i in trange(n_sources, desc='CIS', unit='print', leave=False, dynamic_ncols=True):
            source = dataset[i]['A'].unsqueeze(0).to(device)
            sources = source.expand(batch, -1, -1, -1)
            preds = []
            z = None
            if frozen_style:  # one style for every render of this shoeprint
                z = torch.randn(1, opt.style_dim, device=device).expand(batch, -1)
            for _ in range(math.ceil(n_images / batch)):
                fakes = sb_translate(netG, sources, opt, style=z)  # random style per sample unless frozen
                fakes = F.interpolate(fakes, (299, 299), mode='bicubic',
                                      align_corners=False, antialias=True)
                fakes = fakes.expand(-1, 3, -1, -1)
                fakes = (fakes - fakes.min()) / (fakes.max() - fakes.min() + 1e-14)
                preds.append(F.softmax(inception(fakes), dim=1).cpu().numpy())
            preds = np.concatenate(preds)[:n_images]
            marginal = preds.mean(axis=0)
            kls = [entropy(conditional, marginal) for conditional in preds]
            scores.append(np.exp(np.mean(kls)))

    del inception
    gc.collect()
    torch.cuda.empty_cache()
    return float(np.mean(scores))
