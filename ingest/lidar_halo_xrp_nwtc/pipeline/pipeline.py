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
        dataset["distance"] = (
            "range_gate",
            dataset.coords["range_gate"].data * dataset.attrs["Range gate length (m)"],
        )
        dataset = dataset.swap_dims({"range_gate": "distance"})

        dataset["SNR"].data = 10 * np.log10(dataset.intensity.data - 1)

        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:

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
        location = ds.attrs["location_meaning"]

        # Contour map configurations
        wind_cmap = cmocean.cm.deep_r
        levels = 30

        style_file = os.path.join(os.path.dirname(__file__), "styling.mplstyle")
        with plt.style.context(style_file):

            filename = DSUtil.get_plot_filename(ds, "wind_speed_v_dist_time", "png")
            with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

                fig, axs = plt.subplots(figsize=(14, 8), constrained_layout=True)
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

                # Set the labels and ticks
                fig.suptitle(f"Wind Speed at {location} on {date}")
                add_colorbar(axs, csf, r"Wind Speed (ms$^{-1}$)")
                format_time_xticks(axs)
                axs.set_xlabel("Time (UTC)")
                axs.set_ylabel("Height (m)")

                # Save the figure
                fig.savefig(tmp_path, dpi=100)
                self.storage.save(tmp_path)
                plt.close()

            filename = DSUtil.get_plot_filename(ds, "SNR_v_dist_time", "png")
            with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

                fig, axs = plt.subplots(figsize=(14, 8), constrained_layout=True)

                SNR_v_dist = ds.SNR.where(ds.distance < 5000, drop=True)
                csf = SNR_v_dist.plot.contourf(
                    ax=axs, x="time", levels=levels, cmap=wind_cmap, add_colorbar=False
                )

                # Set the labels and ticks
                fig.suptitle(f"Signal to Noise Ratio at {location} on {date}")
                add_colorbar(axs, csf, "SNR (dB)")
                format_time_xticks(axs)
                axs.set_xlabel("Time (UTC)")
                axs.set_ylabel("Height (m)")

                # Save the figure
                fig.savefig(tmp_path, dpi=100)
                self.storage.save(tmp_path)
                plt.close()
