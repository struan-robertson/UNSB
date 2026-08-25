"""Standalone KID/FID and CIS evaluation of trained checkpoints.

Generates n_evaluation_images translations from the domain-A validation images
(each with a freshly sampled style vector) and scores them against the real
domain-B directory, then computes the conditional inception score (per-input
diversity), mirroring the evaluation of the one_to_many_gan project. The style
and bridge-noise sampling is seeded, so scores are directly comparable across
checkpoints. --epoch all sweeps every numbered checkpoint in ascending order;
--epoch 55-100 restricts the sweep to that inclusive epoch range.

Example:
    python evaluate.py --config config.toml
    python evaluate.py --config config.toml --epoch 200 --n_evaluation_images 10000
    python evaluate.py --config config.toml --epoch all
    python evaluate.py --config config.toml --epoch 55-100
"""
import os
import re
from pathlib import Path

import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model, networks
from util.evaluation import validate_kid_fid, validate_cis, create_image_checkpoints, record_metrics


def load_style_extractor(opt, device):
    """Load the trained style extractor if its checkpoint exists (train-time saves it)."""
    se_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_net_SE.pth' % opt.epoch)
    if not os.path.exists(se_path):
        print('no style extractor checkpoint at %s; skipping the style-transfer image' % se_path)
        return None
    netSE = networks.define_SE(opt.output_nc, opt.style_dim, opt.gpu_ids, opt)
    netSE.load_state_dict(torch.load(se_path, map_location=device, weights_only=True))
    return netSE


def list_epochs(opt):
    """The numbered checkpoints saved for this run, in ascending epoch order."""
    ckpt_dir = Path(opt.checkpoints_dir) / opt.name
    epochs = [int(m.group(1)) for p in ckpt_dir.glob('*_net_G.pth')
              if (m := re.fullmatch(r'(\d+)_net_G', p.stem))]
    return sorted(epochs)


def evaluate_checkpoint(model, opt):
    """FID/KID/CIS plus the visual checkpoints for the currently loaded weights."""
    netSE = load_style_extractor(opt, model.device)
    create_image_checkpoints(model.netG, netSE, opt, model.device, 'epoch_%s' % opt.epoch)

    fid_score, kid_score = validate_kid_fid(model.netG, opt, model.device)
    cis_score = validate_cis(model.netG, opt, model.device) if opt.cond_is_n_evaluation_images > 0 else ''
    record_metrics(opt, fid_score, kid_score, cis_score=cis_score, epoch=opt.epoch)
    print('name: %s, epoch: %s | fid: %.4f, kid: %.6f | cis: %s'
          % (opt.name, opt.epoch, fid_score, kid_score,
             '%.4f' % cis_score if cis_score != '' else 'skipped'))


def main():
    opt = TestOptions().parse()
    # hard-code some parameters, as in test.py
    opt.num_threads = 0
    opt.serial_batches = True
    opt.no_flip = True

    epochs = [opt.epoch]
    if opt.epoch == 'all' or re.fullmatch(r'\d+-\d+', opt.epoch):
        epochs = list_epochs(opt)
        if opt.epoch != 'all':
            lo, hi = map(int, opt.epoch.split('-'))
            epochs = [e for e in epochs if lo <= e <= hi]
        if not epochs:
            raise SystemExit('no numbered checkpoints matching --epoch %s in %s/%s'
                             % (opt.epoch, opt.checkpoints_dir, opt.name))
        print('evaluating %d checkpoints: %s' % (len(epochs), epochs))
        opt.epoch = str(epochs[0])  # setup() below loads the first one

    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    model = create_model(opt)

    # bootstrap the model with one batch so setup() can load the checkpoint
    for data, data2 in zip(dataset, dataset2):
        model.data_dependent_initialize(data, data2)
        model.setup(opt)
        model.parallelize()
        model.eval()
        break

    for i, epoch in enumerate(epochs):
        opt.epoch = str(epoch)
        if i > 0:  # the first checkpoint was already loaded by setup()
            model.load_networks(epoch)
            model.eval()
        evaluate_checkpoint(model, opt)


if __name__ == '__main__':
    main()
