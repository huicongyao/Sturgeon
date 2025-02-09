import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from dataloader import load_numpy_data, ctc_dataset
from models.model import CRF_encoder, ctc_loss
import numpy as np
import time
import pandas as pd
import os
from utils.utils_func import get_logger
from models.decode_utils import *
from tqdm import tqdm
LOGGER = get_logger(__name__)


def train(args):
    st = time.time()
    LOGGER.info("Loading training data")
    train_data, valid_data = load_numpy_data(args.directory, args.split)
    train_data_set = ctc_dataset(train_data)
    valid_data_set = ctc_dataset(valid_data)
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data_set,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=16,
                                               pin_memory=True)
    LOGGER.info("Load data finished!")
    model = CTC_encoder(n_hid=512).cuda()
    if args.load_previous is not None:
        LOGGER.info("Loading previous model")
        model.load_state_dict(torch.load(args.load_previous))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CTCLoss()
    # criterion = ctc_loss
    # save_tag = args.save_tag
    step_interval = int((len(train_loader) + 4) * args.step_rate)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)
    model.train()
    total_tlosses, tlosses, vlosses, vaccs, step_vloss, step_vacc = [], [], [], [], [], []

    scalar = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epochs):
        for batch_idx, (datas, targets, target_lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            # use torch automatic mixed precision to accelerate training speed
            with torch.autocast(device_type='cuda'):
                datas, targets, target_lengths = datas.cuda(), targets.cuda(), target_lengths.cuda()
                inputs = model(datas)
                input_lengths = torch.full(size=(inputs.shape[1],),
                                          fill_value=inputs.shape[0],
                                          dtype=torch.long).cuda()
                # loss = ctc_loss(inputs, targets, target_lengths)
                loss = criterion(inputs, targets, input_lengths, target_lengths)

            if torch.isnan(loss):
                print("NaN loss detected! Skipping this batch.")
                continue  # Skip this batch
            scalar.scale(loss).backward()
            # loss.backward()
            scalar.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # optimizer.step()
            scalar.step(optimizer)
            scalar.update()
            tlosses.append(loss.detach().cpu().numpy()[()])
            if (batch_idx + 1) % step_interval == 0 or (batch_idx + 1) == len(train_loader):
                # code for valid
                model.eval()
                vlosses, vaccs = [], []
                with torch.no_grad():
                    for _, (datas, targets, target_lengths) in enumerate(valid_loader):
                        datas, targets, target_lengths = datas.cuda(), targets.cuda(), target_lengths.cuda()
                        inputs = model(datas)
                        input_lengths = torch.full(size=(inputs.shape[1],),
                                                  fill_value=inputs.shape[0],
                                                  dtype=torch.long).cuda()
                        loss = criterion(inputs, targets, input_lengths, target_lengths)
                        seqs_, traces, quals = viterbi_decode(inputs.cpu())
                        seqs = ["".join(alphabet[x] for x in seq if x != 0) for seq in seqs_]
                        refs = [decode_ref(target, alphabet) for target in targets]
                        accs = [
                            accuracy(ref, seq, min_coverage=0.3) if len(seq) else 0. for ref, seq in zip(refs, seqs)
                        ]
                        vaccs += accs
                        vlosses.append(loss.detach().cpu().numpy()[()])
                print("Epoch: {}, step: [{}/{}], train loss: {:4f},  valid loss:{:4f}, mean acc:{:4f}"
                      .format(epoch + 1, batch_idx + 1, len(train_loader),
                                                      np.mean(tlosses), np.mean(vlosses), np.mean(vaccs))
                              )
                step_vloss.append(np.mean(vlosses))
                step_vacc.append(np.mean(vaccs))
                model.train()
                total_tlosses += tlosses
                tlosses = []
        torch.save(model.state_dict(), os.path.join(args.model_save, "{}_epoch:{}_loss:{:4f}_model.pt".format(args.save_tag, epoch + 1, np.mean(vlosses), np.mean(vaccs))))
        scheduler.step()
        tlosses, vlosses, vaccs = [], [], []

    pd.DataFrame(
        data={
            "train_loss_per_batch_{}".format(args.batch_size): total_tlosses,
        },
        index=range(1, len(total_tlosses) + 1)
    ).to_csv(os.path.join(args.model_save, "{}_training_detail_epoch{}.csv".format(args.save_tag, epoch)))

    pd.DataFrame(
        data = {
            "step_vloss" : step_vloss,
            "step_vacc" : step_vacc,
        }
    ).to_csv(os.path.join(args.model_save, "{}_valid_per_step.csv".format(args.save_tag)))

    LOGGER.info("traing finished, cost: {} seconds!".format(time.time() - st))
    return


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=True
    )

    parser.add_argument("model_save", type=str,
                        help="path to save models and training info")

    parser.add_argument("directory", type=str,
                        help="path to load training data")
    parser.add_argument("--save_tag", type=str, default="CTC",
                        help="tag add to saving file")
    parser.add_argument("--num_epochs", type=int, default=25,
                        help="number of epochs to train")
    parser.add_argument("--step_rate", type=float, default=0.2,
                        help="default steps (total_step * step_rate) to valid")
    parser.add_argument("--split", type=float, default=0.99,
                        help="percentage of data to use for training")
    parser.add_argument("--n_hid", type=int, default=512,
                        help="num of hidden layers for lstm (or d_model for transformer-based)"
                            )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--clip", type=float, default=0.5,
                        help="gradient clipping")
    parser.add_argument("--load_previous", type=str, default=None,
                        help="load previous saved models")

    return parser



if __name__ == "__main__":
    parser = argparser()

    args = parser.parse_args()

    train(args)
