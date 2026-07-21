"""Dataset for the two-directory layout used by the one_to_many_gan project.

Each domain lives in its own root directory containing train/, test/ and val/
subdirectories, with images (searched recursively) inside:

    <dir_A>/train/*.png    <dir_B>/train/*.png
    <dir_A>/test/*.png     <dir_B>/test/*.png
    <dir_A>/val/*.png      <dir_B>/val/*.png

Matching that project's pipeline, images are resized to a fixed size
(crop_size_h x crop_size) with no random cropping, normalised to [-1, 1], and
randomly flipped during training. Grayscale loading is controlled by
input_nc/output_nc. Select with dataset_mode = "twodir".
"""
import random
from pathlib import Path

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from PIL import Image
from data.base_dataset import BaseDataset


class TwoDirDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dir_A', type=str, default=None, help='root directory of domain A (contains train/, test/, val/)')
        parser.add_argument('--dir_B', type=str, default=None, help='root directory of domain B (contains train/, test/, val/)')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        if not opt.dir_A or not opt.dir_B:
            raise ValueError('the twodir dataset requires dir_A and dir_B (set them in the config file or with --dir_A/--dir_B)')

        phase = 'train' if opt.isTrain else opt.phase
        self.A_paths = self._find_images(Path(opt.dir_A).expanduser(), phase)
        self.B_paths = self._find_images(Path(opt.dir_B).expanduser(), phase)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_A = self._build_transform(grayscale=(opt.input_nc == 1))
        self.transform_B = self._build_transform(grayscale=(opt.output_nc == 1))

    def _find_images(self, root, phase):
        phase_dir = root / phase
        if not phase_dir.is_dir() and phase == 'test' and (root / 'val').is_dir():
            phase_dir = root / 'val'
        paths = sorted(list(phase_dir.rglob('*.png')) + list(phase_dir.rglob('*.jpg')))
        if len(paths) == 0:
            raise FileNotFoundError('no .png/.jpg images found under %s' % phase_dir)
        return paths[:min(self.opt.max_dataset_size, len(paths))]

    def _build_transform(self, grayscale):
        transform_list = [transforms.Resize((self.opt.crop_size_h, self.opt.crop_size), InterpolationMode.BICUBIC)]
        if self.opt.isTrain and not self.opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        norm_channels = 1 if grayscale else 3
        transform_list.append(transforms.Normalize((0.5,) * norm_channels, (0.5,) * norm_channels))
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('L' if self.opt.input_nc == 1 else 'RGB')
        B_img = Image.open(B_path).convert('L' if self.opt.output_nc == 1 else 'RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': str(A_path), 'B_paths': str(B_path)}

    def __len__(self):
        """Take the max of the two domain sizes so every image is seen each epoch."""
        return max(self.A_size, self.B_size)
