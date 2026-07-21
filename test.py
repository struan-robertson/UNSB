"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It loads a saved model from --checkpoints_dir and saves the results to --results_dir, as
plain image files under <results_dir>/<name>/<phase>_<epoch>/images/<visual_label>/.

Example:
    python test.py --dataroot ./datasets/horse2zebra --name h2z_SB --mode sb --phase test --eval --num_test 120

See options/base_options.py and options/test_options.py for more test options.
"""
import os

from tqdm import tqdm

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images


def main():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    model = create_model(opt)      # create a model given opt.model and other options

    image_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch), 'images')
    os.makedirs(image_dir, exist_ok=True)
    print('saving results to', image_dir)

    num_test = min(opt.num_test, len(dataset))
    for i, (data, data2) in enumerate(tqdm(zip(dataset, dataset2), total=num_test, unit='img', dynamic_ncols=True)):
        if i == 0:
            model.data_dependent_initialize(data, data2)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data, data2)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        save_images(image_dir, visuals, img_path)


if __name__ == '__main__':
    main()
