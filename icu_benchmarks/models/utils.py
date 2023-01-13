from datetime import timedelta
from enum import Enum
from json import JSONEncoder
import gin
import logging
import numpy as np
import torch


def save_model(model, optimizer, epoch, save_file):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state


def load_model_state(filepath, model, optimizer=None):
    state = torch.load(filepath)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    logging.info("Loaded model and optimizer")


def save_config_file(log_dir):
    config_path = log_dir / "train_config.gin"
    with config_path.open("w") as f:
        f.write(gin.operative_config_str())


class JsonResultLoggingEncoder(JSONEncoder):
    """JSON converter for objects that are not serializable by default."""

    # Serializes foreign datatypes
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, tuple):
            if isinstance(obj)[0] is torch.Tensor or isinstance(obj)[0] is np.ndarray:
                return map(lambda item: item.tolist(), obj)
        if isinstance(obj, timedelta):
            return str(obj)
        return JSONEncoder.default(self, obj)


class Align(Enum):
    LEFT = "<"
    CENTER = "^"
    RIGHT = ">"


def log_table_row(
    cells: list,
    level: int = logging.INFO,
    widths: list[int] = None,
    header: list[str] = None,
    align: Align = Align.LEFT,
    highlight: bool = False,
):
    """Logs a table row.

    Args:
        cells: List of cells to log.
        level: Logging level.
        widths: List of widths for each cell.
        header: List of headers to calculate widths if widths not supplied.
        highlight: If set to true, highlight the row.
    """
    table_cells = cells
    if not widths and header:
        widths = [len(head) for head in header]
    if widths:
        table_cells = []
        for cell, width in zip(cells, widths):
            cell = str(cell)[:width]  # truncate cell if it is too long
            table_cells.append("{: {align}{width}}".format(cell, align=align.value, width=width))
    table_row = " | ".join([f"{cell}" for cell in table_cells])
    if highlight:
        table_row = f"\x1b[31;32m{table_row}\x1b[0m"
    logging.log(level, table_row)
