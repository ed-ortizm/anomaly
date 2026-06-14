"""
Plot spectra, their reconstruction, their residuas
and filters to inspect anomalous behaviors
"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from anomaly.constants import GALAXY_LINES_NM, GALAXY_LINES_NM_NAMES


def add_line_indicators(
    ax,
    wave_nm,
    spec,
    indicator_starts,
    indicator_height,
    add_oii=False,
    add_ne3_he1=False,
    add_h_epsilon=False,
    add_h_delta=False,
    add_h_gamma=False,
    add_oiii=False,
    add_h_beta=False,
    add_sii=False,
    delta=1,
    indicator_color="blue",
    indicator_label_color="black",
    lw=1.5,
    fontsize=8,
):
    """
    Add line indicators to the spectrum plot
    """

    for line_name, value_dict in GALAXY_LINES_NM_NAMES.items():

        # skip desired lines
        if line_name == "OII" and add_oii is False:
            continue

        if line_name == "H_epsilon" and add_h_epsilon is False:
            continue

        if line_name == "H_delta" and add_h_delta is False:
            continue

        if line_name == "H_gamma" and add_h_gamma is False:
            continue

        if line_name == "H_beta" and add_h_beta is False:
            continue

        if line_name == "OIII" and add_oiii is False:
            continue

        if line_name == "SII" and add_sii is False:
            continue

        line_wave = value_dict["line_wave"]
        # Find the flux value of the spectrum at the line's wavelength
        # get small neigborhood among line_wave
        # get max flux there and set as line_flux
        line_flux = np.max(
            spec[(wave_nm > line_wave - delta) & (wave_nm < line_wave + delta)]
        )

        # Plot vertical line indicator
        y_start_indicator = line_flux + indicator_starts
        y_end_indicator = y_start_indicator + indicator_height

        ax.plot(
            [line_wave, line_wave],
            [y_start_indicator, y_end_indicator],
            color=indicator_color,
            lw=lw,
            zorder=3,
        )

        clean_name = value_dict["name"]

        y_text_start = y_end_indicator + indicator_height / 2

        ax.text(
            line_wave,
            y_text_start,
            clean_name,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
            color=indicator_label_color,
            zorder=4,
        )

        # doubles manual setup
        if line_name == "OIII":
            ax.plot(
                [495.9, 495.9],
                [y_start_indicator, y_end_indicator],
                color=indicator_color,
                lw=lw,
                zorder=3,
            )

        if line_name == "H_alpha":

            y_nii_start = y_start_indicator - indicator_height / 2
            y_nii_end = y_end_indicator - indicator_height / 2

            # NII 1st
            ax.plot(
                [654.8, 654.8],
                [y_nii_start, y_nii_end],
                color=indicator_color,
                lw=lw,
                zorder=3,
            )
            y_nii_text = y_nii_end + indicator_height / 2
            ax.text(
                649,
                y_nii_text,
                "NII",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=90,
                color=indicator_label_color,
                zorder=4,
            )

            # NII 2nd
            ax.plot(
                [658.3, 658.3],
                [y_nii_start, y_nii_end],
                color=indicator_color,
                lw=lw,
                zorder=3,
            )

            ax.text(
                664.1,
                y_nii_text,
                "NII",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=90,
                color=indicator_label_color,
                zorder=4,
            )

        if line_name == "SII":
            ax.plot(
                [671.6, 671.6],
                [y_start_indicator, y_end_indicator],
                color=indicator_color,
                lw=lw,
                zorder=3,
            )

    # optionally add NeIII-1 and He I
    if add_ne3_he1 is True:

        line_flux = np.max(spec[(wave_nm > 386.9 - delta) & (wave_nm < 388.9 + delta)])
        y_start_indicator = line_flux + indicator_starts
        y_end_indicator = y_start_indicator + indicator_height

        # add indicator for NeIII 386.9
        ax.plot(
            [386.9, 386.9],
            [y_start_indicator, y_end_indicator],
            color=indicator_color,
            lw=lw,
            zorder=3,
        )
        # add indicator for HI 388.9
        ax.plot(
            [388.9, 388.9],
            [y_start_indicator, y_end_indicator],
            color=indicator_color,
            lw=lw,
            zorder=3,
        )
        # add label NeIII + He I
        y_text_start = y_end_indicator + indicator_height / 2

        ax.text(
            387.9,
            y_text_start,
            "NeIII + HeI",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
            color=indicator_label_color,
            zorder=4,
        )

    return ax


def spec_photo_thumbnail(
    ax,
    wave_nm,
    spec,
    photo_img=None,
    width_height_image="100%",
    bbox_to_anchor=(0.695, 0.52, 0.5, 0.5),
    add_image=False,
) -> tuple:
    """
    Plot spectrum and optionally add a thumbnail of the photo
    """
    ax.plot(wave_nm, spec, color="black")
    # Create inset in top-right corner of figure

    if add_image is False:
        return ax, None

    axins = inset_axes(
        ax,
        width=width_height_image,
        height=width_height_image,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )

    # Show image in the inset
    axins.imshow(photo_img)
    axins.axis("off")  # Hide axis around the image

    return ax, axins


def inspect_reconstruction(
    wave: np.array,
    observation: np.array,
    reconstruction: np.array,
    axs,
):
    """inspect reconstruction"""

    residuals = observation - reconstruction

    for ax in axs:
        ax.clear()

    axs[0].set_ylabel("Median normalized flux")
    axs[1].set_ylabel("Residual")
    axs[1].set_xlabel(r"\lambda [nm]")

    axs[0].plot(wave, observation, c="black", label="observation", lw=1.2)
    axs[0].plot(wave, reconstruction, c="red", label="reconstruction", lw=1.0)

    axs[1].plot(wave, residuals, c="black", lw=1)
    axs[1].hlines(y=0, xmin=wave.min(), xmax=wave.max(), color="blue")

    axs[0].legend()

    max_residuals = np.abs(residuals).max() * 0.5

    axs[1].vlines(
        GALAXY_LINES_NM.values(),
        ymin=-max_residuals,
        ymax=max_residuals,
        color="blue",
        lw=1.5,
    )
