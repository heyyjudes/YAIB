import gin
import lightgbm
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from icu_benchmarks.models.wrappers import MLWrapper
from icu_benchmarks.contants import RunMode


class LGBMWrapper(MLWrapper):
    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fitting function for LGBM models."""
        self.model.fit(
            train_data,
            train_labels,
            eval_set=(val_data, val_labels),
            verbose=True,
            callbacks=[
                lightgbm.early_stopping(self.hparams.patience, verbose=False),
                lightgbm.log_evaluation(period=-1, show_stdv=False),
            ],
        )
        val_loss = list(self.model.best_score_["valid_0"].values())[0]
        return val_loss


@gin.configurable
class LGBMClassifier(LGBMWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lightgbm.LGBMClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class LGBMRegressor(LGBMWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lightgbm.LGBMRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)


# Scikit-learn models
@gin.configurable
class LogisticRegression(MLWrapper):
    __supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LogisticRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable()
class LinearRegression(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LinearRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable()
class ElasticNet(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.ElasticNet, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class RFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(ensemble.RandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class SVMClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.model_args(svm.SVC, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class SVMRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.model_args(svm.SVR, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class PerceptronClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class MLPClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


class MLPRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)
