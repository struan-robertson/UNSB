"""TOML configuration loading.

Mirrors the config.toml approach of the one_to_many_gan project: options live in
a sectioned TOML file instead of a wall of CLI flags. Section names are purely
organisational — every key is flattened onto the single options namespace — with
two exceptions: the [training] section is only applied by train.py and the
[test] section only by test.py, so one file can configure both scripts.

Values from the file override the built-in option defaults; anything passed
explicitly on the command line overrides the file. Note that boolean flags
declared with store_true (e.g. serial_batches) can be switched on from the
file but not switched back off from the command line.
"""
import argparse
import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path('config.toml')


def find_config_path(args):
    """Return the config file to use: --config if given, else ./config.toml if it exists."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    known, _ = pre.parse_known_args(args)
    if known.config is not None:
        path = Path(known.config)
        if not path.exists():
            raise FileNotFoundError('config file not found: %s' % path)
        return path
    return DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None


def load_config_values(args, is_train):
    """Load the TOML config into a flat {option_name: value} dict.

    Parameters:
        args (list or None) -- command-line arguments to scan for --config (None: use sys.argv)
        is_train (bool)     -- selects which of the [training]/[test] sections applies
    """
    path = find_config_path(args)
    if path is None:
        return {}

    with path.open('rb') as f:
        raw = tomllib.load(f)

    skipped_sections = {'test'} if is_train else {'training'}
    values = {}

    def add(key, value, section):
        if key in values:
            raise ValueError("duplicate option '%s' in %s (section [%s])" % (key, path, section))
        values[key] = value

    for key, value in raw.items():
        if isinstance(value, dict):
            if key in skipped_sections:
                continue
            for k, v in value.items():
                add(k, v, key)
        else:
            add(key, value, 'top level')

    return values
