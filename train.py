import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
# from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt

def format_duration(seconds):
    """Converts time in seconds to a formatted string of hours, minutes, and seconds."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))



if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    print(opt.dataroot)
    val_opt = get_val_opt()

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    # train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    # val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    # Loop over the specified number of iterations (epochs)

    # Start time for the total training process
    total_start_time = time.time()

    for epoch in range(opt.niter):
        epoch_start_time = time.time()  # Record the start time of the epoch
        print("Epoch {} started at {}".format(epoch, time.strftime("%H:%M:%S", time.localtime(epoch_start_time))))

        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))

            if model.total_steps % opt.save_latest_freq == 0:
                print('Saving the latest model "{}" (epoch {}, total_steps {})'.format(
                    opt.name, epoch, model.total_steps))
                model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('Saving the model at the end of epoch {}, total iterations {}'.format(
                epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        print("(Validation @ epoch {}) Accuracy: {}; Average Precision: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate reduced. Continuing training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping triggered. Exiting training loop.")
                break

        model.train()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        running_time = epoch_end_time - total_start_time
        print("Epoch {} ended. Duration: {} Running time so far: {}".format(
            epoch, format_duration(epoch_duration), format_duration(running_time)))

    # Print total running time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print("Total running time: {}".format(format_duration(total_duration)))

    # End of training loop

