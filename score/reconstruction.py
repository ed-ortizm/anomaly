"""Get reconstruction based anomaly scores in parallel"""
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import glob
import os
import time

import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import numpy as np

from anomaly import parallelScore
from sdss.utils.managefiles import FileDirectory
from sdss.utils.configfile import ConfigurationFile

def main():
    # Set environment variables to disable multithreading
    # as users will probably want to set the number of cores
    # to the max of their computer.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser(
        description="Train a VAE using config file."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="reconstruction.ini",
        help="Path to config file"
    )

    args = parser.parse_args()

    config_path = args.config
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_path)
    # seed = parser.getint("hyperparaneters", "seed", fallback=0)
    # np.random.seed(seed)
    # tf.random.set_seed(seed)
    #########################################################################
    mp.set_start_method("spawn", force=True)
    #########################################################################
    start_time = time.perf_counter()
    ########################################################################
    configuration = ConfigurationFile()
    ########################################################################
    # Check files and directory
    check = FileDirectory()
    ###########################################################################
    # Load data
    print("Load observations")

    counter = mp.Value("i", 0)
    ###########################################################################
    bin_data_directory = parser.get("directory", "bin_data")
    observation_name = parser.get("file", "observation")
    observation = np.load(f"{bin_data_directory}/{observation_name}")
    share_observation = RawArray(
        np.ctypeslib.as_ctypes_type(observation.dtype), observation.reshape(-1)
    )

    observation_shape = observation.shape
    del observation

    ###########################################################################
    data_directory = parser.get("directory", "data")
    wave_name = parser.get("file", "grid")
    wave = np.load(f"{data_directory}/{wave_name}")
    share_wave = RawArray(np.ctypeslib.as_ctypes_type(wave.dtype), wave)

    del wave
    ###########################################################################
    print("Track meta data", end="\n")

    specobj_ids_name = parser.get("file", "specobjid")
    specobj_ids = np.load(f"{bin_data_directory}/{specobj_ids_name}")

    specobj_id = specobj_ids[:, 1]
    share_specobj_id = RawArray(
        np.ctypeslib.as_ctypes_type(specobj_id.dtype), specobj_id
    )
    del specobj_id

    train_id = specobj_ids[:, 0]
    share_train_id = RawArray(
        np.ctypeslib.as_ctypes_type(train_id.dtype), train_id
    )
    del train_id

    ###########################################################################
    share_model_directory = parser.get("directory", "model")
    check.check_directory(share_model_directory, exit_program=True)

    output_directory = parser.get("directory", "output")
    check.check_directory(output_directory, exit_program=False)

    score_runs = glob.glob(f"{output_directory}/[0-9]*[0-9]/")

    if len(score_runs) == 0:

        run = "00"

    else:

        runs = [int(run.split("/")[-2]) for run in score_runs]
        run = f"{max(runs)+1:02d}"

    output_directory = f"{output_directory}/{run}"
    check.check_directory(f"{output_directory}", exit_program=False)
    ###########################################################################
    # Define grid for anomaly score function
    score_config = parser.items("score")
    score_config = configuration.section_to_dictionary(
        score_config, [",", "\n"]
    )
    parameters_grid = parallelScore.get_grid(score_config)
    ###########################################################################
    number_processes = parser.getint("configuration", "jobs")
    cores_per_worker = parser.getint("configuration", "cores_per_worker")

    parser_directory = os.getcwd()

    with mp.Pool(
        processes=number_processes,
        initializer=parallelScore.init_shared_data,
        initargs=(
            counter,
            share_wave,
            share_observation,
            observation_shape,
            share_specobj_id,
            share_train_id,
            share_model_directory,
            output_directory,
            cores_per_worker,
            config_path,
            parser_directory,
        ),
    ) as pool:

        pool.starmap(parallelScore.compute_anomaly_score, parameters_grid)

    ###########################################################################
    finish_time = time.perf_counter()
    print(f"\n Run time: {finish_time - start_time:.2f}")

if __name__ == "__main__":

    main()
