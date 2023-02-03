import gin
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .constants import DataSegment as Segment, DataSplit as Split


@gin.configurable("Dataset")
class SICUDataset(Dataset):
    """Standardized ICU Dataset: subclass of Torch Dataset that represents the data to learn on.

    Args:
        data: Dict of the different splits of the data.
        split: Either 'train','val' or 'test'.
        vars: Contains the names of columns in the data.
    """

    def __init__(self, data: dict, split: str = Split.train, vars: dict[str] = gin.REQUIRED):
        self.split = split
        self.vars = vars
        self.outcome_df = data[split][Segment.outcome].set_index(self.vars["GROUP"])
        self.features_df = (
            data[split][Segment.features].set_index(self.vars["GROUP"]).drop(labels=self.vars["SEQUENCE"], axis=1)
        )

        # calculate basic info for the data
        self.num_stays = self.outcome_df.index.unique().shape[0]
        self.num_measurements = self.features_df.shape[0]
        self.maxlen = self.features_df.groupby([self.vars["GROUP"]]).size().max()

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.num_stays

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        pad_value = 0.0
        stay_id = self.outcome_df.index.unique()[idx]  # [self.vars["GROUP"]]

        # slice to make sure to always return a DF
        window = self.features_df.loc[stay_id:stay_id].to_numpy()
        labels = self.outcome_df.loc[stay_id:stay_id]["label"].to_numpy(dtype=float)

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate([np.empty(window.shape[0] - 1) * np.nan, labels], axis=0)

        length_diff = self.maxlen - window.shape[0]

        pad_mask = np.ones(window.shape[0])

        # Padding the array to fulfill size requirement
        if length_diff > 0:
            # window shorter than the longest window in dataset, pad to same length
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels = np.concatenate([labels, np.ones(length_diff) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros(length_diff)], axis=0)

        not_labeled = np.argwhere(np.isnan(labels))
        if len(not_labeled) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        data = window.astype(np.float32)

        return torch.from_numpy(data), torch.from_numpy(labels), torch.from_numpy(pad_mask)

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """
        counts = self.outcome_df["label"].value_counts()
        return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(self) -> tuple[np.array, np.array]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A tuple containing data points and label for the split.
        """
        labels = self.outcome_df["label"].to_numpy().astype(float)
        rep = self.features_df
        if len(labels) == self.num_stays:
            # order of groups could be random, we make sure not to change it
            rep = rep.groupby(level=self.vars["GROUP"], sort=False).last()
        rep = rep.to_numpy()

        return rep, labels
