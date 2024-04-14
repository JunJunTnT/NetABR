import argparse
import os
import time
import warnings
from pensieve import Pensieve
from simulator.schedulers import (
    UDRTrainScheduler,
)
from simulator.utils import load_traces
from simulator.abr_trace import AbrTrace

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="./res",
        help="directory to save the model.",
    )
    parser.add_argument("--seed", type=int, default=20, help="seed")
    parser.add_argument(
        "--total-epoch",
        type=int,
        default=500000,
        help="Total number of epoch to be trained.",
    )

    parser.add_argument(
        "--video-size-file-dir",
        type=str,
        default="./data/abr/video_sizes/",
        help="Path to video size files.",
    )

    parser.add_argument(
        "--val-trace-dir",
        default="./data/abr/val_FCC",   # training set
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--time-slot",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device-type",
        default=[0, 0, 1, 1, 2, 2, 3, 3],
    )

    parser.add_argument(
        "--pretrained-abr",
        default=False,
    )

    parser.add_argument(
        "--pretrained-abr-dir",
        default='./res/pretrained_model',
    )

    parser.add_argument(
        "--pretrained-net",
        default=False,
    )

    parser.add_argument(
        "--pretrained-net-dir",
        default='./res/binets_slot=7_04-14 14-08-29/model_saved/',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    pensieve = Pensieve(train_mode=True)
    all_time, all_bw, all_file_names = load_traces(args.val_trace_dir)
    val_traces = [AbrTrace(t, bw, link_rtt=80, buffer_thresh=60, name=name)
                  for t, bw, name in zip(all_time, all_bw, all_file_names)]
    train_scheduler = UDRTrainScheduler(
        val_traces,
        percent=0.0,
    )

    pensieve.train(
        train_scheduler,
        args.save_dir,
        args.total_epoch,
        args.video_size_file_dir,
        args.k,
        args.time_slot,
        args.device_type,
        args.pretrained_abr,
        args.pretrained_abr_dir,
        args.pretrained_net,
        args.pretrained_net_dir,
    )


if __name__ == "__main__":
    t_start = time.time()
    main()
    print("time used: {:.2f}s".format(time.time() - t_start))
