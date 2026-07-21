import time

from tqdm import tqdm

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


def main():
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    visualizer = Visualizer(opt)   # logs losses and saves sample image grids

    total_iters = 0                # the total number of training iterations
    n_epochs_total = opt.n_epochs + opt.n_epochs_decay
    start_epoch = opt.epoch_count

    if opt.continue_train:
        resumed = model.load_training_state(opt.epoch)
        if resumed is not None:
            start_epoch, total_iters = resumed
            print('resuming from epoch %d (total_iters %d); optimizer and LR state will be restored' % (start_epoch, total_iters))
        else:
            print('no %s_state.pth found: loading weights only, starting at epoch %d with fresh optimizers' % (opt.epoch, start_epoch))

    for epoch in range(start_epoch, n_epochs_total + 1):
        epoch_start_time = time.time()
        epoch_iter = 0             # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_start_iters = total_iters   # a resumed run restarts the epoch from its beginning
        dataset.set_epoch(epoch)

        pbar = tqdm(zip(dataset, dataset2), total=dataset_size // opt.batch_size,
                    desc='Epoch %d/%d' % (epoch, n_epochs_total), unit='batch', dynamic_ncols=True)
        iter_data_time = time.time()
        for i, (data, data2) in enumerate(pbar):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            if epoch == start_epoch and i == 0:
                model.data_dependent_initialize(data, data2)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data, data2)   # unpack data from dataset and apply preprocessing
            model.optimize_parameters()    # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            pbar.set_postfix({k: '%.3f' % v for k, v in losses.items()})

            if total_iters % opt.display_freq < batch_size:   # save a grid of the current visuals
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters)

            if total_iters % opt.print_freq < batch_size:     # append losses to the log file
                t_comp = (time.time() - iter_start_time) / batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq < batch_size:   # cache our latest model every <save_latest_freq> iterations
                tqdm.write('saving the latest model (epoch %d, total_iters %d) [%s]' % (epoch, total_iters, opt.name))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                # a resume restarts this epoch from its beginning (the dataloader
                # position cannot be restored), so point the state at the epoch start
                model.save_training_state(epoch, epoch_start_iters, suffix=save_suffix)

            if opt.evaluation_freq > 0 and total_iters % opt.evaluation_freq < batch_size:
                from util.evaluation import validate_kid_fid, training_state_offloaded, create_image_checkpoints, record_metrics
                create_image_checkpoints(model.netG, model.netSE, opt, model.device, 'iter%08d' % total_iters)
                with training_state_offloaded(model):
                    fid_score, kid_score = validate_kid_fid(model.netG, opt, model.device)
                visualizer.log_message('(epoch: %d, total_iters: %d) fid: %.4f kid: %.6f' % (epoch, total_iters, fid_score, kid_score))
                record_metrics(opt, fid_score, kid_score, epoch=epoch, total_iters=total_iters)

            iter_data_time = time.time()

        tqdm.write('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs_total, time.time() - epoch_start_time))
        model.update_learning_rate()                      # update learning rates at the end of every epoch.

        # save after the LR step so the saved scheduler state includes the completed epoch
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            tqdm.write('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            # state is saved for 'latest' only: it is ~170MB (Adam moments for all
            # nets), and resuming from an older numbered epoch falls back to
            # weights-only loading with a printed notice
            model.save_training_state(epoch + 1, total_iters, suffix='latest')


if __name__ == '__main__':
    main()
