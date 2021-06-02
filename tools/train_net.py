import argparse
import os
import torch
import torch.distributed as dist

from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

from siammot.configs.defaults import cfg
from siammot.data.build_train_data_loader import build_train_data_loader
from siammot.modelling.rcnn import build_siammot
from siammot.engine.trainer import do_train
from siammot.utils.get_model_name import get_model_name
from siammot.engine.tensorboard_writer import TensorboardWriter


try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


parser = argparse.ArgumentParser(description="PyTorch SiamMOT Training")
parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
parser.add_argument("--train-dir", default="", help="training folder where training artifacts are dumped", type=str)
parser.add_argument("--model-suffix", default="", help="model suffix to differentiate different runs", type=str)
parser.add_argument("--local_rank", type=int, default=0)


def train(cfg, train_dir, local_rank, distributed, logger):

    # build model
    model = build_siammot(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True
        )

    arguments = {}
    arguments["iteration"] = 0

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, optimizer,
                                         scheduler, train_dir, save_to_disk
                                         )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = build_train_data_loader(
        cfg,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    tensorboard_writer = TensorboardWriter(cfg, train_dir)

    do_train(model, data_loader, optimizer, scheduler,
             checkpointer, device, checkpoint_period, arguments,
             logger, tensorboard_writer
             )

    return model


def setup_env_and_logger(args, cfg):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    model_name = get_model_name(cfg, args.model_suffix)
    train_dir = os.path.join(args.train_dir, model_name)
    if train_dir:
        mkdir(train_dir)

    logger = setup_logger("siammot", train_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(train_dir, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    return train_dir, logger


def main():
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    train_dir, logger = setup_env_and_logger(args, cfg)

    train(cfg, train_dir, args.local_rank, args.distributed, logger)


if __name__ == "__main__":
    main()
