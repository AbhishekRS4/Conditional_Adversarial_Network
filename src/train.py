import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn

from logger_utils import CSVWriter
from dataset import get_dataset_loader
from model import ImageToImageConditionalGAN

def train_gan(FLAGS):
    if not os.path.isdir(FLAGS.dir_model):
        os.makedirs(FLAGS.dir_model)

    csv_writer = CSVWriter(
        file_name=os.path.join(FLAGS.dir_model, ),
        column_names=["epoch", "loss_gen_gan", "loss_gen_l1", "loss_dis_real", "loss_dis_fake"]
    )

    train_dataset_loader = get_dataset_loader(
        FLAGS.dir_dataset_train, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_gan_model = ImageToImageConditionalGAN()
    cond_gan_model.to(device)
    cond_gan_model.train()

    for epoch in range(1, FLAGS.num_epochs + 1):
        epoch_start_time = time.time()
        for data in train_dataset_loader:
            cond_gan_model.set_input(data)
            cond_gan_model.optimize_params()
        epoch_end_time = time.time()
        losses = cond_gan_model.get_current_losses()
        csv_writer.write_row(
            [
                epoch,
                losses["loss_gen_gan"],
                losses["loss_gen_l1"],
                losses["loss_dis_real"],
                losses["loss_dis_fake"],
            ]
        )
        torch.save(cond_gan_model.state_dict(), os.path.join(FLAGS.dir_model, f"{FLAGS.file_model}_{epoch}.pt"))

def main():
    batch_size = 8
    num_epochs = 100
    file_model = "colorizer_cgan"
    file_logger_train = "train_metrics.csv"
    dir_dataset_train = "/home/abhishek/sample_dataset/train/"
    dir_model = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="batch size to use for training")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="num epochs to train the model")
    parser.add_argument("--file_model", default=file_model,
        type=str, help="file name of the model to be used for saving")
    parser.add_argument("--file_logger_train", default=file_logger_train,
        type=str, help="file name of the logger csv file with train losses")
    parser.add_argument("--dir_dataset_train", default=dir_dataset_train,
        type=str, help="full directory path to train dataset")
    parser.add_argument("--dir_model", default=dir_model,
        type=str, help="full directory path to save model files")

    FLAGS, unparsed = parser.parse_known_args()
    train_gan(FLAGS)

if __name__ == "__main__":
    main()
