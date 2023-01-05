import pytest
import icu_benchmarks.data.preprocess as preprocess
from tests.resources.custom_preprocessor import CustomPreprocessor
import gin
from pathlib import Path


def test_preprocessing_hooks():
    gin.bind_parameter("preprocess.preprocessor", CustomPreprocessor)
    preprocess.preprocess_data(
        data_dir=Path("..\demo_data\mortality24\mimic_demo"),
        file_names={"DYNAMIC": "dyn.parquet", "OUTCOME": "outc.parquet", "STATIC": "sta.parquet"},
        vars={
            "GROUP": "stay_id",
            "SEQUENCE": "time",
            "DYNAMIC": [
                "alb",
                "alp",
                "alt",
                "ast",
                "be",
                "bicar",
                "bili",
                "bili_dir",
                "bnd",
                "bun",
                "ca",
                "cai",
                "ck",
                "ckmb",
                "cl",
                "crea",
                "crp",
                "dbp",
                "fgn",
                "fio2",
                "glu",
                "hgb",
                "hr",
                "inr_pt",
                "k",
                "lact",
                "lymph",
                "map",
                "mch",
                "mchc",
                "mcv",
                "methb",
                "mg",
                "na",
                "neut",
                "o2sat",
                "pco2",
                "ph",
                "phos",
                "plt",
                "po2",
                "ptt",
                "resp",
                "sbp",
                "temp",
                "tnt",
                "urine",
                "wbc",
            ],
            "STATIC": ["age", "sex", "height", "weight"],
        },
        num_folds=2,
        fold_index=0,
    )

