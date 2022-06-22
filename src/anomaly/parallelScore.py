"""process base parallelism to compute reconstruction anomaly scores"""
import itertools
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray


import numpy as np
import tensorflow as tf

from anomaly.distance import DistanceAnomalyScore
from anomaly.reconstruction import ReconstructionAnomalyScore
from anomaly.utils import FilterParameters, ReconstructionParameters
from sdss.utils.managefiles import FileDirectory
from autoencoders.ae import AutoEncoder


def to_numpy_array(array: RawArray, array_shape: tuple = None) -> np.array:
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    if array_shape is not None:
        return array.reshape(array_shape)

    return array


###############################################################################
def init_shared_data(
    share_counter: mp.Value,
    share_wave: RawArray,
    share_observation: RawArray,
    data_shape: tuple,
    share_specobj_id: RawArray,
    share_train_id: RawArray,
    share_model_directory: str,
    share_output_directory: str,
    share_cores_per_worker: int,
    share_parser_name: str,
    share_parser_directory: str,
) -> None:
    """
    Initialize worker to train different AEs

    PARAMETERS

        share_counter:
        share_observation:
        data_shape:
        share_model_directory:
        share_output_directory:

    """
    global counter
    global wave
    global observation
    global specobj_id
    global train_id

    global model_directory
    global output_directory

    global cores_per_worker

    global parser_name
    global parser_directory

    counter = share_counter
    wave = to_numpy_array(share_wave)

    observation = to_numpy_array(share_observation, data_shape)

    specobj_id = to_numpy_array(share_specobj_id)
    train_id = to_numpy_array(share_train_id)

    model_directory = share_model_directory
    output_directory = share_output_directory

    cores_per_worker = share_cores_per_worker

    parser_name = share_parser_name
    parser_directory = share_parser_directory

###############################################################################
def compute_anomaly_score(
    metric: str,
    lines: list,
    velocity_filter: float,
    percentage: int,
    relative: bool,
    epsilon: float,
) -> None:
    """
    PARAMETERS
        metric:
    """
    ###########################################################################
    # set the number of cores to use per model in each worker
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=cores_per_worker,
        inter_op_parallelism_threads=cores_per_worker,
        allow_soft_placement=True,
        device_count={"CPU": cores_per_worker},
    )
    session = tf.compat.v1.Session(config=config)
    ###########################################################################
    # Define reconstruction function
    model = AutoEncoder(reload=True, reload_from=model_directory)

    # Define anomaly score class
    anomaly = ReconstructionAnomalyScore(
        model.reconstruct,
        filter_parameters=FilterParameters(
            wave=wave,
            lines=lines,
            velocity_filter=velocity_filter
        ),
        reconstruction_parameters=ReconstructionParameters(
            percentage=percentage,
            relative=relative,
            epsilon=epsilon
        )
    )

    # define name of score:
    # metric_filter_velocity --> has: rel50, noRel75, ...
    score_name = f"{metric}"

    # if have have to filter
    if velocity_filter != 0:

        score_name = f"{score_name}_filter_{velocity_filter}kms"

    # define name of column that will contain the anomaly in the data_frame
    if relative is True:

        score_name = f"{score_name}_rel{percentage}"

    else:

        score_name = f"{score_name}_noRel{percentage}"

    # define name of array if score is saved
    ###########################################################################
    # Compute anomaly score
    with counter.get_lock():

        print(f"[{counter.value}] Compute {score_name}", end="\r")

        counter.value += 1

    score = anomaly.score(observation, metric)

    score_with_ids = np.hstack(
        (
            specobj_id.reshape(-1, 1),
            train_id.reshape(-1, 1),
            score.reshape(-1, 1),
        )
    )
    save_to = f"{output_directory}/{score_name}"
    FileDirectory().check_directory(save_to, exit_program=False)

    np.save(f"{save_to}/{score_name}.npy", score_with_ids)

    # save config file
    with open(
        f"{parser_directory}/{parser_name}", "r", encoding="utf8"
    ) as config_file:

        config = config_file.read()

    with open(f"{save_to}/{parser_name}", "w", encoding="utf8") as config_file:

        config_file.write(config)

    ###########################################################################
    session.close()


def distance_score(
    metric: str,
    lines: list,
    velocity_filter: float,
) -> None:
    """
    Compute anomaly score based on distance between observation
    and reconstruction. Different to the use of residuals between
    observation and reconstruction
    """
    ###########################################################################
    # set the number of cores to use per model in each worker
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=cores_per_worker,
        inter_op_parallelism_threads=cores_per_worker,
        allow_soft_placement=True,
        device_count={"CPU": cores_per_worker},
    )
    session = tf.compat.v1.Session(config=config)
    ###########################################################################
    # Define reconstruction function
    model = AutoEncoder(reload=True, reload_from=model_directory)

    # Define anomaly score class
    anomaly = DistanceAnomalyScore(
        model.reconstruct,
        filter_parameters=FilterParameters(
            wave=wave,
            lines=lines,
            velocity_filter=velocity_filter
        )
    )

    # define name of score:
    # metric_filter_velocity --> has: rel50, noRel75, ...
    score_name = f"{metric}"

    # if have have to filter
    if velocity_filter != 0:

        score_name = f"{score_name}_filter_{velocity_filter}kms"

    # Compute anomaly score
    with counter.get_lock():

        print(f"[{counter.value}] Compute {score_name}", end="\r")

        counter.value += 1

    score = anomaly.score(observation=observation, metric=metric)

    score_with_ids = np.hstack(
        (
            specobj_id.reshape(-1, 1),
            train_id.reshape(-1, 1),
            score.reshape(-1, 1),
        )
    )
    save_to = f"{output_directory}/{score_name}"
    FileDirectory().check_directory(save_to, exit_program=False)

    np.save(f"{save_to}/{score_name}.npy", score_with_ids)

    # save config file
    with open(
        f"{parser_directory}/{parser_name}", "r", encoding="utf8"
    ) as config_file:

        config = config_file.read()

    with open(f"{save_to}/{parser_name}", "w", encoding="utf8") as config_file:

        config_file.write(config)

    session.close()
###############################################################################
def get_grid(parameters: dict) -> itertools.product:
    """
    PARAMETERS
        parameters:

    OUTPUT
        parameters_grid: iterable with the cartesian product
            of input parameters
    """
    for key, value in parameters.items():

        if isinstance(value, list) is False:

            parameters[key] = [value]

    is_reconstruction = len(
        {"lp", "mad", "mse"}.intersection(parameters["metric"])
    ) != 0

    if is_reconstruction is True:

        grid = itertools.product(
            parameters["metric"],
            [parameters["lines"]],  # I need the whole list of lines
            parameters["velocity"],
            parameters["percentage"],
            parameters["relative"],
            parameters["epsilon"],
        )

    else:

        grid = itertools.product(
            parameters["metric"],
            [parameters["lines"]],  # I need the whole list of lines
            parameters["velocity"],
        )

    return grid
