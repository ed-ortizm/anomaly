import matplotlib.pyplot as plt
import numpy as np
import os

wave = np.load("analysis/wave.npy")
###############################################################################
# filter = "filter"
for filter in ["nofilter"]: # ["filter", "nofilter"]:

    if filter == "filter":
        filter_name = "filter_True_100.0kms"
    else:
        filter_name = "filter_False"

    ###########################################################################
    # relative = "relative"
    for relative in ["relative", "norelative"]:

        if relative == "relative":
            relative_name = "mse_relative_True_percentage_100"
        else:
            relative_name = "mse_relative_False_percentage_100"

        #######################################################################
        # data_type = "normal"
        for data_type in ["normal", "anomaly"]: # , "middle"]:

            observation_name = f"{filter}/{relative}/{data_type}/{data_type}_observation_{relative_name}_{filter_name}.npy"

            observation = np.load(f"analysis/{observation_name}")


            reconstruction_name = f"{filter}/{relative}/{data_type}/{data_type}_reconstruction_{relative_name}_{filter_name}.npy"

            reconstruction = np.load(f"analysis/{reconstruction_name}")

            ###################################################################
            fig = plt.figure(figsize=(10, 5), tight_layout=True)
            ax = fig.add_subplot(111)

            # ax.set_xlabel(f"Wavelength $[\AA]$")
            # ax.set_ylabel(f"Normalize flux")

            save_to=f"images/{filter}/{relative}/{data_type}"

            for i in range(observation.shape[0]):

                ax.set_xlabel(f"Wavelength $[\AA]$")
                ax.set_ylabel(f"Normalize flux")

                ax.plot(wave, observation[i], label="Observation")
                ax.plot(wave, reconstruction[i],
                    label="Reconstruction", alpha=0.7
                )

                ax.legend()

                if os.path.exists(save_to) is False:
                    os.makedirs(save_to)

                fname=f"{i:05d}.png"
                print(fname, end="\r")
                fig.savefig(f"{save_to}/{fname}", transparent=False)

                ax.cla()
            plt.close(fig)
