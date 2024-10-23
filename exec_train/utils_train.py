import os
import json
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.display import print_header
from class_model.model import ModelName
from class_dataloader.dataloader import Train
from exec_train.utils_model import load_checkpoint
from exec_train.utils_log import init_metric_logger
from exec_train.utils_dist import get_rank, get_world_size
from class_dataloader.utils import create_dataloader, create_sampler

# Model Pass
def model_pass(model, batch, configs):
    # Set data to device
    index = torch.tensor(batch[0], dtype=torch.float, device=configs.device)

    # TODO: ADD CODE HERE

    # Train model
    losses = model(index=index, device=configs['train_device'])

    # Total loss
    total_loss = sum(loss for loss in losses.values())
    return total_loss, losses

# Training Loop
def train(epoch, model, dataloader, optimizer, configs):
    # Set model to train
    model.train()

    # Set dataloader to a new random sample (only need if running multi-gpu)
    if configs['distributed']:
        dataloader.sampler.set_epoch(epoch)

    # Initialize MetricLogger
    metric_logger = init_metric_logger(configs)

    # Train epoch
    for i, batch in enumerate(metric_logger.log_every(dataloader, print_freq=50, header='Train Epoch: [{}]'.format(epoch))):
        # Reset gradients
        optimizer.zero_grad()

        # Pass through model
        loss_total, losses = model_pass(model, batch, configs)

        # Backward propagation
        loss_total.backward()
        optimizer.step()

        # Update metrics
        [metric_logger.update(**{key: value.item()}) for key, value in losses.items()]; metric_logger.update(loss_total=loss_total, lr_bert=optimizer.param_groups[0]["lr"])

    # Return averaged statistics
    metric_logger.synchronize_between_processes()
    print(f"Averaged Statistics Training Epoch [{epoch}]:", metric_logger.global_avg())
    return {k: "{:.8f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

# Evaluation loop
def eval(epoch, model, dataloader, configs):
    # Set model to eval
    model.eval()

    # Create loss collectors
    loss_accum = {key: [] for key in configs['loss']}

    # Eval
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Pass through model
            loss_total, losses = model_pass(model, batch, configs)

            # Accumulate losses
            for key in losses:
                loss_accum[key].append(losses[key].item())

    # Return averaged statistics
    val_stats = {key: np.mean(values) for key, values in loss_accum.items() if values}
    loss_total = sum(val_stats.values())
    val_stats['loss_total'] = loss_total
    print(f"Averaged Statistics Validation Epoch [{epoch}]: {'  '.join([f'{name}: {value:.8f}' for name, value in val_stats.items()])}")
    return val_stats

# Setup Model
def setup_model(configs):
    # Initialize start epoch
    start_epoch = 0

    # Initialize model
    print_header("Initialize Model")
    model = ModelName(configs=configs)
    model = model.to(device=configs['train_device'])

    # Initialize optimizer
    print_header("Initialize Optimizer")
    optimizer = torch.optim.AdamW([{'lr': configs['eta']},], weight_decay=configs['weight_decay'])

    # Load checkpoint model
    if configs['use_train_checkpoint']:
        model, checkpoint = load_checkpoint(model, configs['train_checkpoint'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    # Store model without DDP for saving (only for multi-gpu)
    model_without_ddp = model
    if configs['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # Return
    return model, model_without_ddp, optimizer, start_epoch

# Setup data
def setup_data(configs):
    # Initialize dataloader
    print_header("Initialize Dataloader")
    train_data = pd.read_csv(configs['train_path'])
    val_data = pd.read_csv(configs['val_path'])

    # Shuffle data
    train_data = train_data.sample(frac=1, random_state=20050531).reset_index(drop=True)

    # Create dataset
    train_dataset = Train(data=train_data)
    val_dataset = Train(data=val_data)

    # Data setup (multi-gpu or not)
    if configs['distributed']:
        # Create sampler
        num_tasks = get_world_size()
        global_rank = get_rank()

        # Create samplers for Distributed Data Parallelism
        train_sampler = create_sampler(datasets=[train_dataset], shuffles=[True], num_tasks=num_tasks, global_rank=global_rank)
        val_sampler = create_sampler(datasets=[val_dataset], shuffles=[False], num_tasks=num_tasks, global_rank=global_rank)

        # Create dataloaders
        train_dataloader = create_dataloader(datasets=[train_dataset], samplers=train_sampler, batch_size=[configs['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
        val_dataloader = create_dataloader(datasets=[val_dataset], samplers=val_sampler, batch_size=[configs['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    else:
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=False, drop_last=False)

    # Return
    return train_dataloader, val_dataloader

# Early stop check
def early_stop_check(val_stats, model_without_ddp, optimizer, configs, epoch, best_loss, epochs_without_improvement):
    # Patience
    patience = configs['early_stop']

    # Check for improvement
    if val_stats['total_loss'] < best_loss:
        best_loss = val_stats['total_loss']
        epochs_without_improvement = 0
        torch.save({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}, os.path.join(configs['output_dir'], 'best_checkpoint.pth'))
    else:
        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= patience:
        print_header(f"Early stop at {epoch}")
        return True, best_loss, epochs_without_improvement

    # Return
    return False, best_loss, epochs_without_improvement

# Save model and log
def save_log(model_without_ddp, optimizer, configs, epoch, train_stats, val_stats):
    # Save model and log results
    save_obj = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}
    torch.save(save_obj, os.path.join(configs['output_dir'], 'checkpoint_%02d.pth' % epoch))
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': "{:.8f}".format(v) for k, v in val_stats.items()}, 'epoch': epoch}
    with open(os.path.join(configs['output_dir'], "log.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")