import os
import copy
import logging
import gin
import json
import hashlib
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import pickle
import pdb
import random


from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit

from icu_benchmarks.data.preprocessor import Preprocessor, DefaultClassificationPreprocessor
from icu_benchmarks.contants import RunMode
from .constants import DataSplit as Split, DataSegment as Segment, VarType as Var


@gin.configurable("preprocess")
def preprocess_data(
    data_dir: Path,
    file_names: dict[str] = gin.REQUIRED,
    preprocessor: Preprocessor = DefaultClassificationPreprocessor,
    use_static: bool = True,
    vars: dict[str] = gin.REQUIRED,
    seed: int = 42,
    debug: bool = False,
    cv_repetitions: int = 5,
    repetition_index: int = 0,
    cv_folds: int = 5,
    train_size: int = None,
    load_cache: bool = False,
    generate_cache: bool = False,
    fold_index: int = 0,
    pretrained_imputation_model: str = None,
    complete_train: bool = False,
    runmode: RunMode = RunMode.classification,
    hospital_id = None, 
    hospital_id_test = None, 
    eval_only = False,
    max_train = False,
    addition_cap = None, 
) -> dict[dict[pd.DataFrame]]:
    """Perform loading, splitting, imputing and normalising of task data.

    Args:
        use_static: Whether to use static features (for DL models).
        complete_train: Whether to use all data for training/validation.
        runmode: Run mode. Can be one of the values of RunMode
        preprocessor: Define the preprocessor.
        data_dir: Path to the directory holding the data.
        file_names: Contains the parquet file names in data_dir.
        vars: Contains the names of columns in the data.
        seed: Random seed.
        debug: Load less data if true.
        cv_repetitions: Number of times to repeat cross validation.
        repetition_index: Index of the repetition to return.
        cv_folds: Number of folds to use for cross validation.
        train_size: Fixed size of train split (including validation data).
        load_cache: Use cached preprocessed data if true.
        generate_cache: Generate cached preprocessed data if true.
        fold_index: Index of the fold to return.
        pretrained_imputation_model: pretrained imputation model to use. if None, standard imputation is used.

    Returns:
        Preprocessed data as DataFrame in a hierarchical dict with features type (STATIC) / DYNAMIC/ OUTCOME
            nested within split (train/val/test).
    """
    
    cache_dir = data_dir / "cache"
    if not use_static:
        file_names.pop(Segment.static)
        vars.pop(Segment.static)

    dumped_file_names = json.dumps(file_names, sort_keys=True)
    dumped_vars = json.dumps(vars, sort_keys=True)

    cache_filename = f"s_{seed}_r_{repetition_index}_f_{fold_index}_t_{train_size}_d_{debug}"

    logging.log(logging.INFO, f"Using preprocessor: {preprocessor.__name__}")
    preprocessor = preprocessor(use_static_features=use_static, save_cache=data_dir / "preproc" / (cache_filename + "_recipe"))
    if isinstance(preprocessor, DefaultClassificationPreprocessor):
        preprocessor.set_imputation_model(pretrained_imputation_model)

    hash_config = hashlib.md5(f"{preprocessor.to_cache_string()}{dumped_file_names}{dumped_vars}".encode("utf-8"))
    cache_filename += f"_{hash_config.hexdigest()}"
    cache_file = cache_dir / cache_filename

    if load_cache:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                logging.info(f"Loading cached data from {cache_file}.")
                return pickle.load(f)
        else:
            logging.info(f"No cached data found in {cache_file}, loading raw features.")

    # Read parquet files into pandas dataframes and remove the parquet file from memory
    logging.info(f"Loading data from directory {data_dir.absolute()}")
    tv_data = {f: pq.read_table(data_dir / file_names[f]).to_pandas(self_destruct=True) for f in file_names.keys()}
    ts_data = None

    # set fixed size test set
    max_test = 400
    # default if no addition cap specified
    if addition_cap is not None: 
        max_per_hospital = int(addition_cap/0.9)
    else: 
        max_per_hospital = 1666 #1500/0.9 sinze 0.9 of the total data is used for train 
    # filter by hospital id
    if hospital_id: 
        hospital_patient_df = pd.read_csv(os.path.join(data_dir, "patient_hospital.csv"))
        hospitals = hospital_id.split("-")
        patient_list = [] 
        for h in hospitals: 
            patient_list += hospital_patient_df[hospital_patient_df["hospitalid"]==int(h)]["patientunitstayid"].to_list()

        if hospital_id_test:
            
            
            # a test hospital is specified
            if hospital_id_test in hospitals and not eval_only: 
                # get list of patients in test hospital 
                logging.info(f"training and testing on same hospital splitting patients") 
                test_patient_list = hospital_patient_df[hospital_patient_df["hospitalid"]==int(hospital_id_test)]["patientunitstayid"].to_list()
                ts_data = {}
                tv_split = {}
                
                df = tv_data['OUTCOME']
                # filter tv_data by patients in test hospital 
                selected = df[df["stay_id"].isin(test_patient_list)]

                # select 400 patients in test hospital as test set
                pos_class = False
                while pos_class == False: 
                    df_test = selected.sample(n=max_test)
                    tot_pos = df[df["stay_id"].isin(df_test['stay_id'])]['label'].sum()
                    pos_class = tot_pos > 1 
                                    
                df_train_val = selected.drop(df_test.index)
                
                    
                for key in tv_data.keys(): 
                    # save half for test data
                    df = tv_data[key]
                    # filter tv_data by patients in test hospital 
                    
                    ts_data[key] = df[df["stay_id"].isin(df_test['stay_id'])]
                    # save half for training data
                    tv_split[key] = df[df["stay_id"].isin(df_train_val['stay_id'])] 
                # get all the patients not in the test hospital
                if len(hospitals) > 2 and max_train: 
                    max_per_hospital_addition = int((max_train - max_per_hospital)/(0.9*len(hospitals)))
                    logging.info(f"more than 2 hospitals limited to {max_per_hospital_addition} additional hospital")
                else: 
                    max_per_hospital_addition = max_per_hospital
                    
                if len(hospitals) > 1: 
                    train_patient_list = [] 
                    for h in hospitals: 
                        if h != hospital_id_test: 

                            hos_patients = hospital_patient_df[hospital_patient_df["hospitalid"]==int(h)]["patientunitstayid"].to_list()
                            all_patients = tv_data['OUTCOME']['stay_id'].tolist()
                            intersection = pd.Series(sorted(set(hos_patients).intersection(set(all_patients))))
                            if addition_cap is not None: 
                    
                                train_patient_list += intersection.sample(n=max_per_hospital_addition).values.tolist()
                            else: 
                                train_patient_list += hos_patients

                    # limit per hospital of training 
                    if addition_cap is not None: 
                        df = tv_split['OUTCOME']
                        if len(df) > max_per_hospital: 
                            selected = df.sample(n=max_per_hospital) 
                        
                        for key in tv_split.keys(): 
                            temp = tv_split[key]
                            tv_split[key] = temp[temp["stay_id"].isin(selected['stay_id'])]
                        
                    
                    # combine selected list with patients from half of the test hospital
                    for key in tv_data.keys(): 
                        df = tv_data[key]
                        selected = df[df["stay_id"].isin(train_patient_list)]
                        tv_data[key] = pd.concat([tv_split[key], selected], ignore_index=True)
                else: 
                    for key in tv_data.keys(): 
                        tv_data[key] = tv_split[key]

            else: 
                assert(complete_train) # must use all data for training/validation if separate test split is specified 
                # we are not using test hospital in the training set
                test_patient_list = hospital_patient_df[hospital_patient_df["hospitalid"]==int(hospital_id_test)]
                test_patient_list = test_patient_list["patientunitstayid"].to_list()
                # patients in hospital and task
                df = tv_data['OUTCOME']

                task_patient_list = df[df["stay_id"].isin(test_patient_list)]["stay_id"].tolist()
                pos_class = False
                while pos_class == False: 
                    sampled_task_patient_list = random.sample(task_patient_list, max_test)
                    tot_pos = df[df["stay_id"].isin(sampled_task_patient_list)]['label'].sum()
                    pos_class = tot_pos > 1 

                # get sampled test data
                ts_data = {}
                for key in tv_data.keys():
                    df = tv_data[key]
                    ts_data[key] = df[df["stay_id"].isin(sampled_task_patient_list)]

                # training data from training hospitals 
                for key in tv_data.keys(): 
                    df = tv_data[key]
                    tv_data[key] = df[df["stay_id"].isin(patient_list)]
            if eval_only: 
                logging.info(f"eval_only mode on {hospital_id_test}: with {ts_data[key].shape[0]} patients")
            else: 
                logging.info(f"testing on hospital(s) {hospital_id_test}: with {ts_data[key].shape[0]} patients") 
                logging.info(f"training on hospital(s) {hospital_id}: with {tv_data[key].shape[0]} patients") 

            # preprocess ts data
            # ts_data = preprocessor.apply(ts_data, vars)
        # no specific test hospital id specified
        else:   
            for key in tv_data.keys(): 
                df = tv_data[key]
                tv_data[key] = df[df["stay_id"].isin(patient_list)]
            logging.info(f"train/test on hospital(s) {hospital_id}: with {tv_data[key].shape[0]} patients") 
                          
    # Generate the splits
    logging.info("Generating splits.")
    if not complete_train:
        tv_data = make_single_split(
            tv_data,
            vars,
            cv_repetitions,
            repetition_index,
            cv_folds,
            fold_index,
            train_size=train_size,
            seed=seed,
            debug=debug,
            runmode=runmode,
            max_train=max_train,
        )
    else:
        # If full train is set, we use all data for training/validation
        if ts_data: 
            # we have a specified dataset that is outside of the current set
            tv_data = make_train_test(tv_data, ts_data, vars, train_size=0.9, test_size=0.8, seed=seed, runmode=runmode, max_train=max_train)
            
        else: 
            tv_data = make_train_val(tv_data, vars, train_size=0.9, seed=seed, debug=debug, runmode=runmode, max_train=max_train)

    # Apply preprocessing
    tv_data = preprocessor.apply(tv_data, vars)

    # Generate cache
    if generate_cache:
        caching(cache_dir, cache_file, tv_data, load_cache)
    else:
        logging.info("Cache will not be saved.")

    logging.info("Finished preprocessing.")

    return tv_data


def make_train_test(
    data: dict[pd.DataFrame],
    ts_data: dict[pd.DataFrame], 
    vars: dict[str],
    train_size=0.8,
    test_size = 1, 
    seed: int = 42,
    runmode: RunMode = RunMode.classification,
    max_train = None,
) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training and validation sets for fitting a full model.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.
    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    # ID variable
    id = vars[Var.group]

    # Get stay IDs from outcome segment
    stays = pd.Series(data[Segment.outcome][id].unique(), name=id)
    

    # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
    labels = data[Segment.outcome].groupby(id).max()[vars[Var.label]].reset_index(drop=True)

    if train_size:
        train_val = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
    train, val = list(train_val.split(stays, labels))[0]

    if max_train: 
        # shuffled already so we can just take first max_test 
        train = train[:max_train] 
        logging.info(f"truncating train set to {max_train} patients") 
        
    split = {Split.train: stays.iloc[train], Split.val: stays.iloc[val]}

        
    data_split = {}

    for fold in split.keys():  # Loop through splits (train / val / test)
        # Loop through segments (DYNAMIC / STATIC / OUTCOME)
        data_split = {}  # Initialize an empty dictionary to store the split data
        for fold in split.keys():  # Iterate over the folds (e.g., 'train', 'val', 'test')
            data_split[fold] = {
            data_type: data[data_type].merge(split[fold], on=id, how="right", sort=True) for data_type in data.keys()
        }


        
    # subsample test set eval 
    # Get stay IDs from outcome segment
    stays = pd.Series(ts_data[Segment.outcome][id].unique(), name=id)
    stays = stays.sample(frac=test_size, random_state=seed)
    labels = ts_data[Segment.outcome].groupby(id).max()[vars[Var.label]].reset_index(drop=True)
    data_split[Split.test] = ts_data

    return data_split
    
def make_train_val(
    data: dict[pd.DataFrame],
    vars: dict[str],
    train_size=0.8,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
    max_train = None
) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training and validation sets for fitting a full model.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.
    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    # ID variable
    id = vars[Var.group]

    # Get stay IDs from outcome segment
    stays = pd.Series(data[Segment.outcome][id].unique(), name=id)

    if debug:
        # Only use 1% of the data
        stays = stays.sample(frac=0.01, random_state=seed)

    # If there are labels, and the task is classification, use stratified k-fold
    if Var.label in vars and runmode is RunMode.classification:
        # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
        labels = data[Segment.outcome].groupby(id).max()[vars[Var.label]].reset_index(drop=True)

        if train_size:
            train_val = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train, val = list(train_val.split(stays, labels))[0]
    else:
        # If there are no labels, use random split
        train_val = ShuffleSplit(train_size=train_size, random_state=seed)
        train, val = list(train_val.split(stays))[0]

    if max_train: 
        # shuffled already so we can just take first max_test 
        train = train[:max_train] 
        logging.info(f"truncating train set to {max_train} patients") 
    split = {Split.train: stays.iloc[train], Split.val: stays.iloc[val]}

    data_split = {} 

    for fold in split.keys():  # Loop through splits (train / val / test)
        # Loop through segments (DYNAMIC / STATIC / OUTCOME)
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        data_split[fold] = {
            data_type: data[data_type].merge(split[fold], on=id, how="right", sort=True) for data_type in data.keys()
        }

    
    # Maintain compatibility with test split
    data_split[Split.test] = copy.deepcopy(data_split[Split.val])
    return data_split


def make_single_split(
    data: dict[pd.DataFrame],
    vars: dict[str],
    cv_repetitions: int,
    repetition_index: int,
    cv_folds: int,
    fold_index: int,
    train_size: int = None,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
    max_train = None, 
) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training, validation, and test set.

    Args:
        runmode: Run mode. Can be one of the values of RunMode
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        cv_repetitions: Number of times to repeat cross validation.
        repetition_index: Index of the repetition to return.
        cv_folds: Number of folds for cross validation.
        fold_index: Index of the fold to return.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.

    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    # ID variable
    id = vars[Var.group]

    if debug:
        # Only use 1% of the data
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        data[Segment.outcome] = data[Segment.outcome].sample(frac=0.01, random_state=seed)
    # Get stay IDs from outcome segment
    stays = pd.Series(data[Segment.outcome][id].unique(), name=id)

    # If there are labels, and the task is classification, use stratified k-fold
    if Var.label in vars and runmode is RunMode.classification:
        # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
        labels = data[Segment.outcome].groupby(id).max()[vars[Var.label]].reset_index(drop=True)
        if labels.value_counts().min() < cv_folds:
            raise Exception(
                f"The smallest amount of samples in a class is: {labels.value_counts().min()}, "
                f"but {cv_folds} folds are requested. Reduce the number of folds or use more data."
            )
        if train_size:
            outer_cv = StratifiedShuffleSplit(cv_repetitions, train_size=train_size)
        else:
            outer_cv = StratifiedKFold(cv_repetitions, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)

        dev, test = list(outer_cv.split(stays, labels))[repetition_index]
        dev_stays = stays.iloc[dev]
        train, val = list(inner_cv.split(dev_stays, labels.iloc[dev]))[fold_index]
    else:
        # If there are no labels, or the task is regression, use regular k-fold.
        if train_size:
            outer_cv = ShuffleSplit(cv_repetitions, train_size=train_size)
        else:
            outer_cv = KFold(cv_repetitions, shuffle=True, random_state=seed)
        inner_cv = KFold(cv_folds, shuffle=True, random_state=seed)

        dev, test = list(outer_cv.split(stays))[repetition_index]
        dev_stays = stays.iloc[dev]
        train, val = list(inner_cv.split(dev_stays))[fold_index]
    
    if max_train: 
        # shuffled already so we can just take first max_test 
        train = train[:max_train] 
        logging.info(f"truncating train set to {max_train} patients") 
    split = {
        Split.train: dev_stays.iloc[train],
        Split.val: dev_stays.iloc[val],
        Split.test: stays.iloc[test],
    }
    data_split = {}

    for fold in split.keys():  # Loop through splits (train / val / test)
        # Loop through segments (DYNAMIC / STATIC / OUTCOME)
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        data_split[fold] = {
            data_type: data[data_type].merge(split[fold], on=id, how="right", sort=True) for data_type in data.keys()
        }

    return data_split


def caching(cache_dir, cache_file, data, use_cache, overwrite=True):
    if use_cache and (not overwrite or not cache_file.exists()):
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file.touch()
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Cached data in {cache_file}.")
