
# process base parallelism to train models in a grid of hyperparameters
import itertools
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray


import numpy as np

###############################################################################
def to_numpy_array(array: RawArray, array_shape: tuple) -> np.array:
    """Create a numpy array backed by a shared memory Array."""

    array = np.ctypeslib.as_array(array)

    return array.reshape(array_shape)


###############################################################################
def init_shared_data(
    share_counter: mp.Value,
    share_wave: RawArray,
    share_data: RawArray,
    data_shape: tuple,
    data_location: str,
    share_model_directory: str,
    share_output_directory: str,
    share_cores_per_worker: int,
) -> None:
    """
    Initialize worker to train different AEs

    PARAMETERS

        share_counter:
        share_data:
        data_shape:
        data_location:
        share_model_directory:
        share_output_directory:

    """
    global counter
    global wave
    global data

    global model_directory
    global output_directory

    global cores_per_worker
    global session

    counter = share_counter
    wave =  share_wave
    observation = to_numpy_array(share_data, data_shape)
    observation[...] = np.load(data_location)

    model_directory = share_model_directory
    output_directory = share_output_directory

    cores_per_worker = share_cores_per_worker

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
    # Define reconstruction function

    model = AutoEncoder(reload=True, reload_from=model_location)
    reconstruct_function = model.reconstruct

###############################################################################
def compute_anomaly_score(
) -> None:
    """
    PARAMETERS
        metric:
        reconstruct_function:
        # wave: this is shared parameter
        # model_directory: shared parameter
        lines:
        velocity_filter:
        percentage:
        relative:
        epsilon:
    """
    ###########################################################################
    # import tensorflow as tf
    # from autoencoders.ae import AutoEncoder
    #
    # # set the number of cores to use per model in each worker
    # jobs = cores_per_worker
    # config = tf.compat.v1.ConfigProto(
    #     intra_op_parallelism_threads=jobs,
    #     inter_op_parallelism_threads=jobs,
    #     allow_soft_placement=True,
    #     device_count={"CPU": jobs},
    # )
    # session = tf.compat.v1.Session(config=config)
    # ###########################################################################
    # # Define reconstruction function
    # model = AutoEncoder(reload=True, reload_from=model_location)
    # reconstruct_function = model.reconstruct
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

    filter = lines != None

    if filter is True:

        df_name = f"{df_name}_filter_{velocity}Kms"

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

        print(f"Compute {score_name}", end="\r")

        counter.value += 1

    score = anomaly.score(observation, metric)
    np.save_score(f"{output_directory}/{score_name}.npy", score)
    ###########################################################################
    session.close()
###############################################################################
def get_parameters_grid(hyperparameters: dict) -> itertools.product:
    """
    Returns cartesian product of hyperparameters: reconstruction_weight,
        mmd_weights, kld_weights, alpha and lambda

    PARAMETERS
        hyperparameters:

    OUTPUT
        parameters_grid: iterable with the cartesian product
            of input parameters
    """
    for key, value in hyperparameters.items():

        if type(value) != type([]):
            hyperparameters[key] = [value]

    grid = itertools.product(
        hyperparameters["reconstruction_weight"],
        hyperparameters["mmd_weight"],
        hyperparameters["kld_weight"],
        hyperparameters["alpha"],
        hyperparameters["lambda"],
    )

    return grid
