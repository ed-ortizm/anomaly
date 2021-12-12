import matplotlib.pyplot as plt
import numpy as np
import os

def spectrum_plot(wave, flux_1, label_1, flux_2=None, label_2=None, alpha=0.5, fname="test.png", save_to=".", close=True, all=True):

    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.set_xlabel(f"Wavelength $[\AA]$")
    ax.set_ylabel(f"Normalize flux")
    plt.tight_layout()

    if all is True:

        for i in range(flux_1.shape[0]):

            ax.plot(wave, flux_1[i], label=label_1)
            ax.plot(wave, flux_2[i], label=label_2, alpha=alpha)
            ax.legend()

        if os.path.exists(save_to) is False:

            os.makedirs(save_to)

        print(fname)
        fig.savefig(f"{save_to}/{fname}", transparent=True)

wave = np.load("analysis/wave.npy")
###############################################################################
# filter = "filter"
for filter in ["filter", "nofilter"]:

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
        for data_type in ["anomaly"]: #["normal", "anomaly", "middle"]:

            observation_name = f"{filter}/{relative}/{data_type}/{data_type}_observation_{relative_name}_{filter_name}.npy"

            observation = np.load(f"analysis/{observation_name}")


            reconstruction_name = f"{filter}/{relative}/{data_type}/{data_type}_reconstruction_{relative_name}_{filter_name}.npy"

            reconstruction = np.load(f"analysis/{reconstruction_name}")

            ###################################################################
            fig = plt.figure(figsize=(10, 5), tight_layout=True)
            ax = fig.add_subplot(111)

            ax.set_xlabel(f"Wavelength $[\AA]$")
            ax.set_ylabel(f"Normalize flux")

            save_to=f"images/{filter}/{relative}/{data_type}"

            for i in range(observation.shape[0]):

                ax.plot(wave, observation[i], label="Observation")
                ax.plot(wave, reconstruction[i], label="Reconstruction;", alpha=0.7)
                ax.legend()

                if os.path.exists(save_to) is False:
                    os.makedirs(save_to)

                fname=f"{i:05d}.png"
                print(fname)
                fig.savefig(f"{save_to}/{fname}", transparent=False)

                ax.cla()

            plt.close()    
