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
    active_lines=None,
    delta=1,
    indicator_color="blue",
    indicator_label_color="black",
    lw=1,
    fontsize=6,
):
    """
    Add line indicators to the spectrum plot.
    """

    # lines to add
    if active_lines is None:
        active_lines = []

    # plot and text logic
    def _draw_line(wave, label, y_start, y_end, label_x=None, label_y=None):
        ax.plot([wave, wave], [y_start, y_end], color=indicator_color, lw=lw, zorder=3)
        if label:
            ax.text(
                label_x or wave,
                label_y or (y_end + indicator_height / 2),
                label,
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=90,
                color=indicator_label_color,
                zorder=4,
            )

    for line_name, value_dict in GALAXY_LINES_NM_NAMES.items():

        # Skip if the line wasn't requested.
        if line_name not in active_lines:
            continue

        line_wave = value_dict["line_wave"]
        clean_name = value_dict["name"]

        # Calculate local max flux
        mask = (wave_nm > line_wave - delta) & (wave_nm < line_wave + delta)
        line_flux = np.max(spec[mask]) if np.any(mask) else 0

        y_start = line_flux + indicator_starts
        y_end = y_start + indicator_height

        # Draw the primary line
        _draw_line(line_wave, clean_name, y_start, y_end)

        # Handle secondary lines
        if line_name == "OIII":
            _draw_line(495.9, None, y_start, y_end)

        elif line_name == "H_alpha":
            # NII lines get shifted down slightly to prevent overlapping text
            y_nii_start = y_start - indicator_height / 2
            y_nii_end = y_end - indicator_height / 2
            _draw_line(654.8, "NII", y_nii_start, y_nii_end, label_x=649)
            _draw_line(658.3, "NII", y_nii_start, y_nii_end, label_x=664.1)

        elif line_name == "SII":
            _draw_line(671.6, None, y_start, y_end)

    return ax


def spec_photo_thumbnail(
    ax,
    wave_nm,
    spec,
    photo_img=None,
    width_height_image="100%",
    bbox_to_anchor=(0.695, 0.52, 0.5, 0.5),
    add_image=False,
    lw=1.2,
) -> tuple:
    """
    Plot spectrum and optionally add a thumbnail of the photo
    """
    ax.plot(wave_nm, spec, color="black", lw=lw)
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
