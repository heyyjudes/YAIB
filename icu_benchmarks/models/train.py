import os
import gin
import json
import torch
import logging
import pandas as pd
import numpy as np
from joblib import load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC, Accuracy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pathlib import Path
from icu_benchmarks.data.loader import PredictionDataset, ImputationDataset
from icu_benchmarks.models.utils import save_config_file, JSONMetricsLogger
from icu_benchmarks.contants import RunMode
from icu_benchmarks.data.constants import DataSplit as Split
from ignite.contrib.metrics import AveragePrecision, ROC_AUC
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score, precision_recall_curve, balanced_accuracy_score
import pdb

cpu_core_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

def assure_minimum_length(dataset):
    # set max dataset len to 400
    if len(dataset) < 2:
        return [dataset[0], dataset[0]]
    return dataset


@gin.configurable("train_common")
def train_common(
    data: dict[str, pd.DataFrame],
    log_dir: Path,
    eval_only: bool = False,
    load_weights: bool = False,
    source_dir: Path = None,
    reproducible: bool = True,
    mode: str = RunMode.classification,
    model: object = gin.REQUIRED,
    weight: str = None,
    optimizer: type = Adam,
    precision=32,
    batch_size=400,
    epochs=1000,
    patience=20,
    min_delta=1e-5,
    test_on: str = Split.test,
    dataset_names=None,
    use_wandb: bool = False,
    cpu: bool = False,
    verbose=False,
    ram_cache=False,
    pl_model=True,
    train_only=False,
    num_workers: int = min(cpu_core_count, torch.cuda.device_count() * 8 * int(torch.cuda.is_available()), 32),
):
    """Common wrapper to train all benchmarked models.

    Args:
        data: Dict containing data to be trained on.
        log_dir: Path to directory where model output should be saved.
        eval_only: If set to true, skip training and only evaluate the model.
        load_weights: If set to true, skip training and load weights from source_dir instead.
        source_dir: If set to load weights, path to directory containing trained weights.
        reproducible: If set to true, set torch to run reproducibly.
        mode: Mode of the model. Can be one of the values of RunMode.
        model: Model to be trained.
        weight: Weight to be used for the loss function.
        optimizer: Optimizer to be used for training.
        precision: Pytorch precision to be used for training. Can be 16 or 32.
        batch_size: Batch size to be used for training.
        epochs: Number of epochs to train for.
        patience: Number of epochs to wait for improvement before early stopping.
        min_delta: Minimum change in loss to be considered an improvement.
        test_on: If set to "test", evaluate the model on the test set. If set to "val", evaluate on the validation set.
        use_wandb: If set to true, log to wandb.
        cpu: If set to true, run on cpu.
        verbose: Enable detailed logging.
        ram_cache: Whether to cache the data in RAM.
        pl_model: Loading a pytorch lightning model.
        num_workers: Number of workers to use for data loading.
    """

    logging.info(f"Training model: {model.__name__}.")
    dataset_class = ImputationDataset if mode == RunMode.imputation else PredictionDataset

    logging.info(f"Logging to directory: {log_dir}.")
    save_config_file(log_dir)  # We save the operative config before and also after training

    train_dataset = dataset_class(data, split=Split.train, ram_cache=ram_cache, name=dataset_names["train"])
    val_dataset = dataset_class(data, split=Split.val, ram_cache=ram_cache, name=dataset_names["val"])
    train_dataset, val_dataset = assure_minimum_length(train_dataset), assure_minimum_length(val_dataset)
    batch_size = min(batch_size, len(train_dataset), len(val_dataset))

    if not eval_only:
        logging.info(
            f"Training on {train_dataset.name} with {len(train_dataset)} samples and validating on {val_dataset.name} with"
            f" {len(val_dataset)} samples."
        )
    logging.info(f"Using {num_workers} workers for data loading.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_shape = next(iter(train_loader))[0].shape

    if load_weights:
        model = load_model(model, source_dir, pl_model=pl_model)
    else:
        model = model(optimizer=optimizer, input_size=data_shape, epochs=epochs, run_mode=mode)

    model.set_weight(weight, train_dataset)
    model.set_trained_columns(train_dataset.get_feature_names())
    loggers = [TensorBoardLogger(log_dir), JSONMetricsLogger(log_dir)]
    if use_wandb:
        loggers.append(WandbLogger(save_dir=log_dir))
    callbacks = [
        EarlyStopping(monitor="val/loss", min_delta=min_delta, patience=patience, strict=False, verbose=verbose),
        ModelCheckpoint(log_dir, filename="model", save_top_k=1, save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    if verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=min(100, len(train_loader) // 2)))
    if precision == 16 or "16-mixed":
        torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        max_epochs=epochs if model.requires_backprop else 1,
        callbacks=callbacks,
        precision=precision,
        accelerator="auto" if not cpu else "cpu",
        devices=max(torch.cuda.device_count(), 1),
        deterministic="warn" if reproducible else False,
        benchmark=not reproducible,
        enable_progress_bar=verbose,
        logger=loggers,
        num_sanity_val_steps=-1,
        log_every_n_steps=5,
    )
    
    if not eval_only:
        if model.requires_backprop:
            logging.info("Training DL model.")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            logging.info("Training complete.")
        else:
            logging.info("Training ML model.")
            model.fit(train_dataset, val_dataset)
            model.save_model(log_dir, "last")
            logging.info("Training complete.")
    if train_only:
        logging.info("Finished training full model.")
        save_config_file(log_dir)
        return 0
    
    test_dataset = dataset_class(data, split=test_on, name=dataset_names["test"])
    test_dataset = assure_minimum_length(test_dataset)
    logging.info(f"Testing on {test_dataset.name}  with {len(test_dataset)} samples.")
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=min(batch_size * 8, len(test_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if model.requires_backprop
        else DataLoader([test_dataset.to_tensor()], batch_size=1)
    )

    # test_loader = DataLoader([test_dataset.to_tensor()], batch_size=1)
    model.set_weight("balanced", train_dataset)
    test_loss = trainer.test(model, dataloaders=test_loader, verbose=verbose)[0]["test/loss"]
    # additional metrics 

    #threshold for accuracy score 
    
    if not model.requires_backprop: 
        # select threshold for non-DL models
        val_rep, val_label = val_dataset.get_data_and_labels()
        val_pred = model.predict(val_rep)[:, 1]
        precision, recall, thresholds = precision_recall_curve(val_label, val_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        thresh = thresholds[ix]

    fold_metrics = {} 
    test_rep, test_label, test_sex, test_race = test_dataset.get_data_and_labels(groups=True)    
    # compute overall accuracy
    if model.requires_backprop: 
        for batch in test_loader: 
            data, labels, mask = batch
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float()
            else:
                data = data.float()
            
            out = model(data)
            
            # Get prediction and target
            prediction = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1])
            target = torch.masked_select(labels, mask)
            
            transformed_output = model.output_transform((prediction, target))
            auc_fn = AUROC(task="binary")
            fold_metrics['AUC_TEST'] = auc_fn(transformed_output[0], transformed_output[1])
            
            bacc_fn = Accuracy(task="multiclass", num_classes=2, average='macro')
            fold_metrics['BACC_TEST'] = bacc_fn(transformed_output[0]>0.5, transformed_output[1])

            acc_fn = Accuracy(task="binary")
            acc_fn(transformed_output[0]>0.5, transformed_output[1])


    else: 
        test_pred = model.predict(test_rep)[:, 1]
        auc_fn = roc_auc_score
        acc_fn = accuracy_score
        bacc_fn = balanced_accuracy_score

        fold_metrics['ACC_TEST'] = acc_fn(test_label, test_pred>thresh)
        fold_metrics['BACC_TEST'] = bacc_fn(test_label, test_pred>thresh)
        fold_metrics['AUC_TEST'] = auc_fn(test_label, test_pred)

    vals, cnts = np.unique(test_sex, return_counts=True) 

    for v, c in zip(vals, cnts): 
        mask = test_sex == v
        mask = mask.values
        gv = len(np.unique(test_label[mask]))
        if gv > 1: 
            # ordering of score and label is different between torch metrics and sklearn
            if model.requires_backprop: 
                fold_metrics[f'gender{v}_AUC_TEST'] = auc_fn(transformed_output[0][mask], transformed_output[1][mask])
                fold_metrics[f'gender{v}_BACC_TEST'] = bacc_fn(transformed_output[0][mask]>0.5, transformed_output[1][mask])
                fold_metrics[f'gender{v}_ACC_TEST'] = acc_fn(transformed_output[0][mask]>0.5, transformed_output[1][mask])
            else: 
                fold_metrics[f'gender{v}_AUC_TEST'] = auc_fn(test_label[mask], test_pred[mask])
                fold_metrics[f'gender{v}_BACC_TEST'] = bacc_fn(test_label[mask], test_pred[mask]>thresh)
                fold_metrics[f'gender{v}_ACC_TEST'] = acc_fn(test_label[mask], test_pred[mask]>thresh)

    # race
    vals, cnts = np.unique(test_race, return_counts=True) 
    for v, c in zip(vals, cnts):  
        mask = test_race == v
        mask = mask.values
        gv = len(np.unique(test_label[mask]))
        if gv > 1: 
            if model.requires_backprop: 
                fold_metrics[f'race{v}_AUC_TEST'] = auc_fn(transformed_output[0][mask], transformed_output[1][mask])
                fold_metrics[f'race{v}_BACC_TEST'] = bacc_fn(transformed_output[0][mask]>0.5, transformed_output[1][mask])
                fold_metrics[f'race{v}_ACC_TEST'] = acc_fn(transformed_output[0][mask]>0.5, transformed_output[1][mask])
            else: 
                fold_metrics[f'race{v}_AUC_TEST'] = auc_fn(test_label[mask], test_pred[mask])
                fold_metrics[f'race{v}_BACC_TEST'] = bacc_fn(test_label[mask], test_pred[mask]>thresh)
                fold_metrics[f'race{v}_ACC_TEST'] = acc_fn(test_label[mask], test_pred[mask]>thresh)


    if model.requires_backprop: 
        for key in fold_metrics: 
            fold_metrics[key] = fold_metrics[key].item()
            
    with open(os.path.join(log_dir, "additional_metrics.json"), "w") as outfile: 
        json.dump(fold_metrics, outfile)

        
    save_config_file(log_dir)
    return test_loss


def load_model(model, source_dir, pl_model=True):
    if source_dir.exists():
        if model.requires_backprop:
            if (source_dir / "model.ckpt").exists():
                model_path = source_dir / "model.ckpt"
            elif (source_dir / "model-v1.ckpt").exists():
                model_path = source_dir / "model-v1.ckpt"
            elif (source_dir / "last.ckpt").exists():
                model_path = source_dir / "last.ckpt"
            else:
                return Exception(f"No weights to load at path : {source_dir}")
            if pl_model:
                model = model.load_from_checkpoint(model_path)
            else:
                checkpoint = torch.load(model_path)
                model.load_from_checkpoint(checkpoint)
        else:
            model_path = source_dir / "model.joblib"
            model = load(model_path)
    else:
        raise Exception(f"No weights to load at path : {source_dir}")
    logging.info(f"Loaded {type(model)} model from {model_path}")
    return model
