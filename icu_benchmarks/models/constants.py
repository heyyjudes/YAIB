from ignite.contrib.metrics import AveragePrecision, ROC_AUC, PrecisionRecallCurve, RocCurve
from ignite.metrics import Accuracy, RootMeanSquaredError  # , ConfusionMatrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    roc_curve,
    # confusion_matrix,
    r2_score,
    mean_squared_error,
    # f1_score,
)
from torchmetrics.classification import BinaryFairness, AUROC, AveragePrecision, \
    PrecisionRecallCurve, ROC, CalibrationError, F1Score
from enum import Enum
from icu_benchmarks.models.custom_metrics import CalibrationCurve, BalancedAccuracy, MAE, JSD, TorchMetricsWrapper, BinaryFairnessWrapper


# TODO: revise transformation for metrics in wrappers.py in order to handle metrics that can not handle a mix of binary and
#  continuous targets
class MLMetrics:
    BINARY_CLASSIFICATION = {
        "AUC": roc_auc_score,
        "Calibration_Curve": calibration_curve,
        "PR": average_precision_score,
        "PR_Curve": precision_recall_curve,
        "RO_Curve": roc_curve,
    }

    MULTICLASS_CLASSIFICATION = {
        "Accuracy": accuracy_score,
        "AUC": roc_auc_score,
        "Balanced_Accuracy": balanced_accuracy_score,
        "PR": average_precision_score,
    }

    REGRESSION = {
        "MAE": mean_absolute_error,
        "R2": r2_score,
        "RMSE": mean_squared_error,
    }


# TODO: add support for confusion matrix
class DLMetrics:
    BINARY_CLASSIFICATION = {
        "AUC": ROC_AUC,
        "Calibration_Curve": CalibrationCurve,
        "PR": AveragePrecision,
        "PR_Curve": PrecisionRecallCurve,
        "RO_Curve": RocCurve,
    }

    BINARY_CLASSIFICATION_TORCHMETRICS = {
        "AUC": AUROC(task="binary"),
        "PR": AveragePrecision(task="binary"),
        "Calibration_Error": CalibrationError(task="binary", n_bins=10),
        "F1": F1Score(task="binary", num_classes=2),
        "Binary_Fairness": BinaryFairnessWrapper(num_groups=2, task='demographic_parity', group_name="sex"),
    }

    MULTICLASS_CLASSIFICATION = {
        "Accuracy": Accuracy,
        "BalancedAccuracy": BalancedAccuracy,
    }

    REGRESSION = {
        "MAE": MAE,
    }

    IMPUTATION = {
        "rmse": RootMeanSquaredError,
        "mae": MAE,
        "jsd": JSD,
    }


class ImputationInit(str, Enum):
    """Type of initialization to use for the imputation model."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    XAVIER = "xavier"
    KAIMING = "kaiming"
    ORTHOGONAL = "orthogonal"
