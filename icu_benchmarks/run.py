# -*- coding: utf-8 -*-
from datetime import datetime

import gin
import logging
import sys
from pathlib import Path
import importlib.util

from icu_benchmarks.tuning.hyperparameters import choose_and_bind_hyperparameters
from scripts.plotting.utils import plot_aggregated_results
from icu_benchmarks.cross_validation import execute_repeated_cv
from icu_benchmarks.run_utils import (
    build_parser,
    create_run_dir,
    aggregate_results,
    log_full_line,
)


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)
    # Whether to enable verbose logging. If disabled reduces log output desired for running compute cluster jobs.
    gin.bind_parameter("DLWrapper.verbose_logging", args.verbose)

    load_weights = args.command == "evaluate"
    name = args.name
    task = args.task
    model = args.model
    experiment = args.experiment
    source_dir = None
    # todo:check if this is correct
    reproducible = False
    log_dir_name = args.log_dir / name
    log_dir = (log_dir_name / experiment) if experiment else (log_dir_name / args.task_name / model)
    train_on_cpu = args.cpu

    if args.preprocessor:
        # Import custom supplied preprocessor
        try:
            spec = importlib.util.spec_from_file_location("CustomPreprocessor", args.preprocessor)
            module = importlib.util.module_from_spec(spec)
            sys.modules["preprocessor"] = module
            spec.loader.exec_module(module)
            gin.bind_parameter("preprocess.preprocessor", module.CustomPreprocessor)
        except Exception as e:
            logging.error(f"Could not import custom preprocessor from {args.preprocessor}: {e}")
    else:
        from icu_benchmarks.data.preprocessor import DefaultPreprocessor as preprocessor

    if train_on_cpu:
        gin.bind_parameter("DLWrapper.device", "cpu")
    if load_weights:
        # Evaluate
        log_dir /= f"from_{args.source_name}"
        run_dir = create_run_dir(log_dir)
        source_dir = args.source_dir
        gin.parse_config_file(source_dir / "train_config.gin")
        if gin.query_parameter("train_common.model").selector == "DLWrapper":
            # Calculate input dimensions for deep learning models based on preprocessing operations
            gin.bind_parameter("model/hyperparameter.input_dim", preprocessor().calculate_input_dim())

    else:
        # Train
        reproducible = args.reproducible
        checkpoint = log_dir / args.checkpoint if args.checkpoint else None
        gin_config_files = (
            [Path(f"configs/experiments/{args.experiment}.gin")]
            if args.experiment
            else [Path(f"configs/models/{model}.gin"), Path(f"configs/tasks/{task}.gin")]
        )
        gin.parse_config_files_and_bindings(gin_config_files, args.hyperparams, finalize_config=False)

        if gin.query_parameter("train_common.model").selector == "DLWrapper":
            # Calculate input dimensions for deep learning models based on preprocessing operations
            gin.bind_parameter("model/hyperparameter.input_dim", preprocessor().calculate_input_dim())

        run_dir = create_run_dir(log_dir)
        choose_and_bind_hyperparameters(
            args.tune,
            args.data_dir,
            run_dir,
            args.seed,
            checkpoint=checkpoint,
            debug=args.debug,
            generate_cache=args.generate_cache,
            load_cache=args.load_cache,
        )

    logging.info(f"Logging to {run_dir.resolve()}")
    log_full_line("STARTING TRAINING", level=logging.INFO, char="=", num_newlines=3)
    start_time = datetime.now()
    execute_repeated_cv(
        args.data_dir,
        run_dir,
        args.seed,
        load_weights=load_weights,
        source_dir=source_dir,
        reproducible=reproducible,
        debug=args.debug,
        load_cache=args.load_cache,
        generate_cache=args.generate_cache,
    )

    log_full_line("FINISHED TRAINING", level=logging.INFO, char="=", num_newlines=3)
    execution_time = datetime.now() - start_time
    log_full_line(f"DURATION: {execution_time}", level=logging.INFO, char="")
    aggregate_results(run_dir, execution_time)
    if args.plot:
        plot_aggregated_results(run_dir, "aggregated_test_metrics.json")


"""Main module."""
if __name__ == "__main__":
    main()
