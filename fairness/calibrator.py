from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Tuple, List
from sklearn.isotonic import IsotonicRegression


def balanced_subsample(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a balanced subsample of the data by sampling equal numbers of positive and negative examples.
    
    Parameters
    ----------
    y_true : np.ndarray
        Binary true targets
    y_prob : np.ndarray
        Raw probability/score of the positive class
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Balanced subsample of y_true and y_prob
    """
    # Get indices of positive and negative examples
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    # Determine the number of samples to take from each class
    n_samples = min(len(pos_indices), len(neg_indices))
    
    # Randomly sample equal numbers from each class
    np.random.seed(42)  # For reproducibility
    pos_sampled = np.random.choice(pos_indices, size=n_samples, replace=False)
    neg_sampled = np.random.choice(neg_indices, size=n_samples, replace=False)
    
    # Combine the sampled indices
    sampled_indices = np.concatenate([pos_sampled, neg_sampled])
    np.random.shuffle(sampled_indices)  # Shuffle to mix positive and negative examples
    
    return y_true[sampled_indices], y_prob[sampled_indices]

def create_binned_data(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       n_bins: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Bin ``y_true`` and ``y_prob`` by distribution of the data.
    i.e. each bin will contain approximately an equal number of
    data points. Bins are sorted based on ascending order of ``y_prob``.

    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.

    y_prob : 1d ndarray
        Raw probability/score of the positive class.

    n_bins : int, default 15
        A bigger bin number requires more data.

    Returns
    -------
    binned_y_true/binned_y_prob : 1d ndarray
        Each element in the list stores the data for that bin.
    """
    sorted_indices = np.argsort(y_prob)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_prob = y_prob[sorted_indices]
    binned_y_true = np.array_split(sorted_y_true, n_bins)
    binned_y_prob = np.array_split(sorted_y_prob, n_bins)
    return binned_y_true, binned_y_prob


def get_bin_boundaries(binned_y_prob: List[np.ndarray]) -> np.ndarray:
    """
    Given ``binned_y_prob`` from ``create_binned_data`` get the
    boundaries for each bin.

    Parameters
    ----------
    binned_y_prob : list
        Each element in the list stores the data for that bin.

    Returns
    -------
    bins : 1d ndarray
        Boundaries for each bin.
    """
    bins = []
    for i in range(len(binned_y_prob) - 1):
        last_prob = binned_y_prob[i][-1]
        next_first_prob = binned_y_prob[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)

    bins.append(1.0)
    return np.array(bins)

def compute_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute the Expected Calibration Error (ECE) between predicted probabilities and true outcomes.
    
    Parameters
    ----------
    y_true : np.ndarray
        Binary true targets
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int, default=10
        Number of bins to use for computing calibration error
        
    Returns
    -------
    float
        Expected Calibration Error
    """
    # Create bins of equal width
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    
    # Initialize arrays to store bin statistics
    bin_counts = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    
    # Compute statistics for each bin
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_counts[i] = np.sum(mask)
            bin_accuracies[i] = np.mean(y_true[mask])
            bin_confidences[i] = np.mean(y_prob[mask])
    
    # Compute ECE
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(y_true)
    
    return ece

class Calibrator:
    def __init__(self, name:str):
        self.params = {}
        self.calibrator_fn = None
        self.name = name


class HistogramCalibrator(Calibrator):
    def __init__(self, name:str = "HistogramCalibrator", bins=20):
        super().__init__(name)
        self.name = name
        self.n_bins = bins

    def calibrate(self, y_prob:np.array, y_true:np.array, subsample=False):
        # insert subsample code here
        if subsample:
            y_true, y_prob = balanced_subsample(y_true=y_true,
                                                   y_prob=y_prob)

        binned_y_true, binned_y_prob = create_binned_data(y_true, y_prob, self.n_bins)
        self.bins_ = get_bin_boundaries(binned_y_prob)
        self.bins_score_ = np.array([np.mean(value) for value in binned_y_true])

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        '''
        :param y_prob: Array of uncalibrated prediction probabilities, typically between 0 and 1
        :return: Array of calibrated probabilities matched to their corresponding bins
        '''
        indices = np.searchsorted(self.bins_, y_prob)
        return self.bins_score_[indices]


class PlattCalibrator(Calibrator):
    def __init__(self):
        self.name = 'PlattScaling'
        super().__init__("PlattCalibrator")

    def calibrate(self, y_prob: np.ndarray, y_true: np.ndarray, subsample=False):
        if subsample:
            y_true, y_prob = balanced_subsample(y_true=y_true,
                                                   y_prob=y_prob)
        logistic = LogisticRegression(C=1, solver='lbfgs')
        #logistic = LogisticRegression()
        logistic.fit(y_prob.reshape(-1, 1), y_true)
        coeff = logistic.coef_[0]
        intercept = logistic.intercept_
        self.params['coeff'] = coeff
        self.params['intercept'] = intercept
        return

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        '''
        :param y_prob: Array of uncalibrated prediction probabilities, typically between 0 and 1
        :return: Array of calibrated probabilities matched to their corresponding bins
        '''
        out = y_prob * self.params['coeff'] + self.params['intercept']
        return 1/ (1+ np.exp(-out))


class BinningCalibrator(Calibrator):
    def __init__(self, bins=20):
        self.name = 'BinningCalibrator'
        self.B = bins
        super().__init__("BinningCalibrator")

    def calibrate(self, y_prob: np.ndarray, y_true : np.ndarray, subsample=False):
        if subsample:
            y_true, y_prob = balanced_subsample(y_true=y_true,
                                                   y_prob=y_prob)
        sorted_y = np.asarray(y_true)[np.argsort(y_prob)]
        scores = np.asarray(y_prob)[np.argsort(y_prob)]
        binned_y_true = np.array_split(sorted_y, self.B)
        binned_y_prob = np.array_split(scores, self.B)

        intervals = []
        new_values = []

        for i in range(len(binned_y_prob) - 1):
            curr_chunk = binned_y_prob[i]
            next_chunk = binned_y_prob[i + 1]
            intervals.append(np.mean([curr_chunk[-1], next_chunk[0]]))
            curr_true = binned_y_true[i]
            new_values.append(np.mean(curr_true))
        intervals.append(1.0)
        new_values.append(np.mean(binned_y_true[-1]))

        self.params['intervals'] = np.asarray(intervals)
        self.params['new_values'] = np.asarray(new_values)
        return

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        '''
        :param y_prob: Array of uncalibrated prediction probabilities, typically between 0 and 1
        :return: Array of calibrated probabilities matched to their corresponding bins
        '''
        indices = np.searchsorted(self.params['intervals'], y_prob)
        return self.params['new_values'][indices]

class IsotonicCalibrator(Calibrator):
    def __init__(self):
        self.name = 'IsotonicCalibrator'
        super().__init__("IsotonicCalibrator")
        self.isotonic_regressor = None

    def calibrate(self, y_prob: np.ndarray, y_true: np.ndarray, subsample=False):
        """
        Fit an isotonic regression model to calibrate the probabilities.
        
        Parameters
        ----------
        y_prob : np.ndarray
            Raw probability/score of the positive class
        y_true : np.ndarray
            Binary true targets
        subsample : bool, default=False
            Whether to use balanced subsampling
        """
        if subsample:
            y_true, y_prob = balanced_subsample(y_true=y_true,
                                               y_prob=y_prob)
        
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(y_prob, y_true)

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Transform uncalibrated probabilities using the fitted isotonic regression model.
        
        Parameters
        ----------
        y_prob : np.ndarray
            Array of uncalibrated prediction probabilities
            
        Returns
        -------
        np.ndarray
            Array of calibrated probabilities
        """
        if self.isotonic_regressor is None:
            raise ValueError("Model has not been calibrated yet. Call calibrate() first.")
        return self.isotonic_regressor.predict(y_prob)
   
