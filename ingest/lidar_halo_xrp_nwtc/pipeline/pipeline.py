import os
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Dict
from tsdat import DSUtil
from utils import A2ePipeline


class Pipeline(A2ePipeline):
    """--------------------------------------------------------------------------------
    LIDAR HALO XRP NWTC INGESTION PIPELINE

    Ingest for the XRP Halo Lidar at the NWTC site

    --------------------------------------------------------------------------------"""

    def hook_customize_dataset(
        self, dataset: xr.Dataset, raw_mapping: Dict[str, xr.Dataset]
    ) -> xr.Dataset:
        # Compress row of variables in input into variables dimensioned by time and height
        for raw_filename, raw_dataset in raw_mapping.items():
            # convert range gate to distance and change coords
            if ".hpl" in raw_filename:
                dataset["distance"] = (
                    "range_gate",
                    dataset.coords["range_gate"].data
                    * dataset.attrs["Range gate length (m)"],
                )
                dataset = dataset.swap_dims({"range_gate": "distance"})

                dataset["SNR"] = 10 * np.log10(dataset.intensity - 1)

        return dataset

    def hook_generate_and_persist_plots(self, dataset: xr.Dataset):
        def format_time_xticks(ax, start=4, stop=21, step=4, date_format="%H-%M"):
            ax.xaxis.set_major_locator(
                mpl.dates.HourLocator(byhour=range(start, stop, step))
            )
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(date_format))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

        def add_colorbar(ax, plot, label):
            cb = plt.colorbar(plot, ax=ax, pad=0.01)
            cb.ax.set_ylabel(label, fontsize=12)
            cb.outline.set_linewidth(1)
            cb.ax.tick_params(size=0)
            cb.ax.minorticks_off()
            return cb

        ds = dataset
        date = pd.to_datetime(ds.time.data[0]).strftime("%d-%b-%Y")

        # Colormaps to use
        wind_cmap = cmocean.cm.deep_r
        avail_cmap = cmocean.cm.amp_r

        style_file = os.path.join(os.path.dirname(__file__), "styling.mplstyle")
        with plt.style.context(style_file):

            filename = DSUtil.get_plot_filename(
                dataset, "wind_speed_v_dist_time", "png"
            )
            with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

                # Calculations for contour plots
                levels = 30

                # Create figure and axes objects
                fig, axs = plt.subplots(
                    nrows=1, figsize=(14, 8), constrained_layout=True
                )
                fig.suptitle(f"Wind Speed at {ds.attrs['location_meaning']} on {date}")

                # Make top subplot -- contours and quiver plots for wind speed and direction
                los_wind_speed = ds.doppler.where(ds.distance < 5000, drop=True)
                csf = los_wind_speed.plot.contourf(
                    ax=axs,
                    x="time",
                    levels=levels,
                    cmap=wind_cmap,
                    add_colorbar=False,
                    vmin=-5,
                    vmax=5,
                )
                add_colorbar(axs, csf, r"Wind Speed (ms$^{-1}$)")

                # # Make bottom subplot -- heatmap for data availability
                # da = ds.data_availability.plot(ax=axs[1], x="time", cmap=avail_cmap, add_colorbar=False, vmin=0, vmax=100)
                # add_colorbar(axs[1], da, "Availability (%)")

                # Set the labels and ticks
                # for i in range(1):
                format_time_xticks(axs)
                axs.set_xlabel("Time (UTC)")
                axs.set_ylabel("Height (m)")

                # Save the figure
                fig.savefig(tmp_path, dpi=100)
                self.storage.save(tmp_path)
                plt.close()

            filename = DSUtil.get_plot_filename(dataset, "SNR_v_dist_time", "png")
            with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

                # Calculations for contour plots
                levels = 30

                # Create figure and axes objects
                fig, axs = plt.subplots(
                    nrows=1, figsize=(14, 8), constrained_layout=True
                )
                fig.suptitle(
                    f"Signal to Noise Ratio at {ds.attrs['location_meaning']} on {date}"
                )

                # Make top subplot -- contours and quiver plots for wind speed and direction
                SNR_v_dist = ds.SNR.where(ds.distance < 5000, drop=True)
                csf = SNR_v_dist.plot.contourf(
                    ax=axs, x="time", levels=levels, cmap=wind_cmap, add_colorbar=False
                )
                add_colorbar(axs, csf, "SNR (dB)")

                # # Make bottom subplot -- heatmap for data availability
                # da = ds.data_availability.plot(ax=axs[1], x="time", cmap=avail_cmap, add_colorbar=False, vmin=0, vmax=100)
                # add_colorbar(axs[1], da, "Availability (%)")

                # Set the labels and ticks
                # for i in range(1):
                format_time_xticks(axs)
                axs.set_xlabel("Time (UTC)")
                axs.set_ylabel("Height (m)")

                # Save the figure
                fig.savefig(tmp_path, dpi=100)
                self.storage.save(tmp_path)
                plt.close()
