"""
Plot spectra, their reconstruction, their residuas
and filters to inspect anomalous behaviors
"""
import matplotlib.pyplot as plt
import numpy as np

from anomaly.constants import GALAXY_LINES


def inspect_reconstruction(
    wave: np.array,
    observation: np.array,
    reconstruction: np.array,
    fig,
    axs,
    format: str,
    save_to: str,
    rank: int,
):

    residuals = observation - reconstruction

    for ax in axs: ax.clear() 


    axs[0].set_ylabel("Median normalized flux")
    axs[1].set_ylabel("Residual")
    axs[1].set_xlabel("$\lambda [\AA]$")

    axs[0].plot(wave, observation, c="black", label="observation", lw=1.2)
    axs[0].plot(wave, reconstruction, c="red", label="reconstruction", lw=1.)

    axs[1].plot(wave, residuals, c="black", lw=1)
    axs[1].hlines(y=0, xmin=wave.min(), xmax=wave.max(), color="blue")


    axs[0].legend()

    max_residuals = np.abs(residuals).max() * 0.5

    axs[1].vlines(
        GALAXY_LINES.values(),
        ymin=-max_residuals,
        ymax=max_residuals,
        color="blue",
        lw=1.5
    )

    fig.savefig(f"{save_to}/rank_{rank:04d}.{format}")
    fig.savefig(f"{save_to}/rank_{rank:04d}.pdf")
