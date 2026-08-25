# Style-conditioned Schrödinger bridge

Translation of shoeprints to shoemarks with a Schrödinger bridge conditioned on
a style vector, so one shoeprint maps to many plausible shoemarks. Forked from
the [Unpaired Neural Schrödinger Bridge](https://github.com/cyclomon/UNSB)
implementation of Kim et al. (ICLR 2024) and extended with style conditioning
by weight modulation, a decoupled bridge-noise multiplier, and the evaluation
and analysis tooling described below.

## Installation

Choose the build of torch that matches the card:

```sh
uv sync --extra cuda
uv sync --extra rocm
```

## Configuration

Options resolve as defaults, then a TOML configuration file, then command-line
arguments. Options live on a single flat namespace and are declared in
`options/`; sections in the file are organisational only, except `[training]`,
which is applied by `train.py` alone, and `[test]`, applied by `test.py` and by
inference through `unsb_handler.py`. An unknown key in the file is an error.
Boolean options accept an explicit value, so anything set in the file can be
turned off again on the command line.

```sh
uv run python train.py --config config.toml
uv run python train.py --config config.toml --batch_size 2 --no_flip false
```

`config.toml` trains the stochastic bridge. `config_no_noise.toml` sets
`bridge_noise = 0` for the deterministic bridge adopted in the thesis; the
multiplier must match between training and inference.

## Evaluation

```sh
uv run python evaluate.py --config config_no_noise.toml --epoch 56
uv run python evaluate.py --config config_no_noise.toml --epoch all
```

Generates seeded translations of the source images and scores them against the
real target domain with clean-fid, reporting FID, KID and the conditional
inception score. Results are appended to `metrics.csv` in the run's checkpoint
directory. `--epoch all` sweeps every numbered checkpoint, and a range such as
`--epoch 55-100` restricts the sweep.

## Synthetic shoemark pools

`generate_pool.py` writes a pool of synthetic shoemarks per shoeprint class for
training the retrieval model in the siamese project, which streams them from
disk rather than generating during training. It supports the `unsb`, `gan` and
`munit` backends so that all three generators supply data through one path.

```sh
uv run python generate_pool.py --backend unsb --n-styles 657
```

`style_coverage.py` measures how many styles per shoeprint are needed to cover
a generator's output diversity, which sets the pool size for each run.

## Analysis

`style_space_unsb.py` examines the geometry of the style space,
`nfe_analysis.py` the effect of the number of bridge steps, `style_per_step.py`
where in the trajectory style takes effect, and `qualitative_unsb.py` produces
the figure grids.
