"""
@file main.py

Main entrypoint for the training and testing environments. Takes in a configuration file
of arguments and either trains a model or tests a given model and checkpoint.
"""
import shutil
import argparse
import pytorch_lightning

from torch.utils.data import DataLoader
from utils.dataloader import BaseDataset
from utils.utils import parse_args, get_exp_versions, strtobool, find_best_epoch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


if __name__ == '__main__':
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(125125125, workers=True)

    # Define the parser with the configuration file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/CONFIG_EXAMPLE.json',
                        help='path and name of the configuration .json to load')
    parser.add_argument('--train', type=strtobool, default=True,
                        help='whether to train or test the given model')
    parser.add_argument('--resume', type=strtobool, default=False,
                        help='whether to continue training from the checkpoint in the config')

    # Parse the config file and get the model function used
    args, model_type = parse_args(parser)

    # Get version numbers
    top, exptop = get_exp_versions(args.model, args.exptype)

    # Build datasets
    train_dataset = BaseDataset(file_path=f"data/{args.dataset}/{args.dataset_ver}/train.npz", config=args)
    test_dataset = BaseDataset(file_path=f"data/{args.dataset}/{args.dataset_ver}/test.npz", config=args)

    # Build dataloaders
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, drop_last=True)

    # Initialize model
    model = model_type(args, top, exptop)

    # Callbacks for checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_recon_mse',
                                          filename='epoch{epoch:02d}-val_recon_mse{val_recon_mse:.4f}',
                                          auto_insert_metric_name=False, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val_recon_mse", min_delta=0.00001, patience=10, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pytorch_lightning.Trainer.from_argparse_args(
        args,
        callbacks=[                     # Which callbacks to use
            early_stop_callback,
            lr_monitor,
            checkpoint_callback
        ],
        deterministic=True,             # Deterministic call on PyTorch to enable cross-device reproducibility
        max_epochs=args.num_epochs,     # Total number of epochs to run
        gradient_clip_val=5.0,          # Gradient clipping, specifically stabilizes early Neural SSM training
        check_val_every_n_epoch=10,     # How often to do the validation check
        num_sanity_val_steps=0,         # Just simply removing some pre-training checks that affect saving
        auto_select_gpus=True
    )

    # Check whether to train, resume training from a checkpoint, or test
    if args.train is True and args.resume is False:
        trainer.fit(model, train_dataloader, test_dataloader)

    # If resuming training, get the last ckpt if not given one
    elif args.train is True and args.resume is True:
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/last.ckpt"

        trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)

    # Otherwise test the model, using the best epoch if none is given
    else:
        ckpt_path = args.ckpt_path + "/checkpoints/" + args.checkpt if args.checkpt != "None" \
            else f"{args.ckpt_path}/checkpoints/{find_best_epoch(args.ckpt_path)[0]}"

        trainer.test(model, test_dataloader, ckpt_path=ckpt_path)

    # After running the given task, delete the last generated lightning_logs
    shutil.rmtree(f"lightning_logs/version_{top}")
