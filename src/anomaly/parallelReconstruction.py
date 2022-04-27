# process base parallelism to train models in a grid of parameters
import itertools
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray


import numpy as np

from anomaly.reconstruction import ReconstructionAnomalyScore
from sdss.superclasses import FileDirectory
###############################################################################
def to_numpy_array(array: RawArray, array_shape: tuple=None) -> np.array:
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    if array_shape != None:
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
    wave =  to_numpy_array(share_wave)

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
    metric,
    lines,
    velocity_filter,
    percentage,
    relative,
    epsilon,
) -> None:
    """
    PARAMETERS
        metric:
    """
    ###########################################################################
    import tensorflow as tf
    from autoencoders.ae import AutoEncoder

    # set the number of cores to use per model in each worker
    jobs = cores_per_worker
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=jobs,
        inter_op_parallelism_threads=jobs,
        allow_soft_placement=True,
        device_count={"CPU": jobs},
    )
    session = tf.compat.v1.Session(config=config)
    ###########################################################################
    # Define reconstruction function
    model = AutoEncoder(reload=True, reload_from=model_directory)
    reconstruct_function = model.reconstruct
    ###########################################################################
    # Define anomaly score class
    anomaly = ReconstructionAnomalyScore(
        reconstruct_function,
        wave,
        lines=lines,
        velocity_filter=velocity_filter,
        percentage=percentage,
        relative=relative,
        epsilon=epsilon,
    )

    # define name of score:
    # metric_filter_velocity --> has: rel50, noRel75, ...
    df_name = f"{metric}"

    filter = velocity_filter != 0

    if filter is True:

        df_name = f"{df_name}_filter_{velocity_filter}Kms"

    # define name of column that will contain the anomaly in the data_frame
    column_df_name = f"{percentage}"

    if relative is True:

        column_df_name = f"rel{column_df_name}"

    else:

        column_df_name = f"noRel{column_df_name}"

    # define name of array if score is saved
    score_name = f"{df_name}_{column_df_name}"
    ###########################################################################
    # Compute anomaly score
    with counter.get_lock():

        print(f"[{counter.value}] Compute {score_name}", end="\r")

        counter.value += 1

    score = anomaly.score(observation, metric)

    score_with_ids = np.hstack(
        (
            specobj_id.reshape(-1,1),
            train_id.reshape(-1,1),
            score.reshape(-1,1))
    )
    save_to = f"{output_directory}/{df_name}"
    FileDirectory().check_directory(save_to, exit=False)

    np.save(f"{save_to}/{score_name}.npy", score_with_ids)

    # save config file
    with open(f"{parser_location}/{parser_name}", "r") as config_file:

        config = config_file.read()

    with open(f"{save_to}/{parser_name}", "w") as config_file:

        config_file.write(config)


    ###########################################################################
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

        if type(value) != type([]):

            parameters[key] = [value]

    grid = itertools.product(
        parameters["metric"],
        [parameters["lines"]], # I need the whole list of lines
        parameters["velocity"],
        parameters["percentage"],
        parameters["relative"],
        parameters["epsilon"],
    )

    return grid
