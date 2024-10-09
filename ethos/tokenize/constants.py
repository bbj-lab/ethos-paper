from dataclasses import dataclass
from enum import Enum

import numpy as np

TOKEN_DTYPE = np.uint16
TIME_DTYPE = np.float64

SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60


class Dataset(Enum):
    MIMIC = "mimic"


class DataFold(Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass
class DataProp:
    name: Dataset
    fold: DataFold
    dataset_dir: str
    id_col: str
    fold_dir: str
    fold_suffix: str
    csv_format: str
    module: str

    @staticmethod
    def create(dataset_name: str, fold_name: str):
        dataset = Dataset(dataset_name)
        fold = DataFold(fold_name)

        suffix = "Data"
        suffix += "Training" if fold == DataFold.TRAIN else "Test"
        return DataProp(
            name=dataset,
            fold=fold,
            dataset_dir="mimic-iv-3.0_Data",
            id_col="subject_id",
            fold_dir=f"mimic-iv-3.0_{suffix}", # f"{DataFold.TRAIN}"
            fold_suffix=f"", # f"_{fold.value}"
            csv_format="csv.gz",
            module=dataset.value,
        )
