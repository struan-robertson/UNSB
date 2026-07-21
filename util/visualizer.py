"""Lightweight training/test visualisation.

Replaces the old visdom/dominate stack with tqdm-friendly console logging,
a plain-text loss log, and sample image grids saved to disk.
"""
import os
import time

from PIL import Image, ImageDraw
from tqdm import tqdm

from . import util


def save_images(image_dir, visuals, image_path, aspect_ratio=1.0):
    """Save images to the disk.

    Parameters:
        image_dir (str)          -- directory to save into; one subdirectory per visual label
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy)) pairs
        image_path (str list)    -- the path of the source image, used to name the outputs
        aspect_ratio (float)     -- the aspect ratio of saved images
    """
    short_path = os.path.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, label, '%s.png' % name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)


class Visualizer():
    """Prints/saves losses (console via tqdm.write, plus loss_log.txt) and saves
    the current visuals as a single labelled image grid under
    <checkpoints_dir>/<name>/samples/.
    """

    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.sample_dir = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
        util.mkdirs([self.sample_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    CAPTION_HEIGHT = 20

    def display_current_results(self, visuals, epoch, total_iters):
        """Save the current visuals as one horizontally-tiled grid image,
        with each tile's label in a black caption bar above it."""
        tiles = []
        for label, image in visuals.items():
            img = Image.fromarray(util.tensor2im(image))
            tile = Image.new('RGB', (img.width, img.height + self.CAPTION_HEIGHT), (0, 0, 0))
            tile.paste(img, (0, self.CAPTION_HEIGHT))
            draw = ImageDraw.Draw(tile)
            draw.text((5, 4), label, fill=(255, 255, 255))
            tiles.append(tile)
        if not tiles:
            return

        w = max(t.width for t in tiles)
        h = max(t.height for t in tiles)
        grid = Image.new('RGB', (w * len(tiles), h))
        for i, tile in enumerate(tiles):
            grid.paste(tile, (i * w, 0))
        grid.save(os.path.join(self.sample_dir, 'epoch%03d_iter%08d.png' % (epoch, total_iters)))

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """Print current losses on console (without breaking tqdm bars) and append to the log file.

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        tqdm.write(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def log_message(self, message):
        """Print an arbitrary message on console (tqdm-safe) and append it to the log file."""
        tqdm.write(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
