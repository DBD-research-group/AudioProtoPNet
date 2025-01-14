from typing import List, Tuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection, PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


class RadarAxes(PolarAxes):
    """
    Custom axes class for radar plots, extending the PolarAxes class from
    Matplotlib. It provides functionality specifically for creating radar charts
    (spider charts, star plots).

    Attributes:
        name (str): Name of the axis, set to 'radar'.
        frame (str): Shape of the radar chart's frame, either 'circle' or 'polygon'.
        num_vars (int): Number of variables represented in the radar chart.
        theta (numpy.ndarray): Array of angles for the variables.
    """

    name = "radar"

    def __init__(self, *args, frame: str = "polygon", num_vars: int = 5, **kwargs):
        """
        Initialize RadarAxes with a specific frame shape and number of variables.

        Args:
            *args: Arguments passed to the PolarAxes initialization.
            frame (str): Shape of the radar chart's frame ('circle' or 'polygon').
            num_vars (int): Number of variables in the radar chart.
            **kwargs: Keyword arguments passed to the PolarAxes initialization.
        """
        self.frame = frame
        self.num_vars = num_vars
        super().__init__(*args, **kwargs)
        self.theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        self.set_theta_zero_location("N")

    def fill(self, *args, closed: bool = True, **kwargs):
        """
        Fill the radar plot, ensuring the filled areas are closed shapes.

        Args:
            *args: Variable length argument list.
            closed (bool): Whether to close the filled shape.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list: A list of matplotlib.patches.Patch objects for the filled area.
        """
        return super().fill(closed=closed, *args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Plot data on the radar axes, ensuring lines are closed.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list: A list of Line2D objects for the plotted data.
        """
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)
        return lines

    def _close_line(self, line):
        """
        Close the line by connecting its end to the start.

        Args:
            line (matplotlib.lines.Line2D): The line to be closed.
        """
        x, y = line.get_data()
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.concatenate((x, [x[0]]))
            y = np.concatenate((y, [y[0]]))
            line.set_data(x, y)

    def set_varlabels(self, labels, label_fontsize: int):
        """
        Set variable labels at the specified angles.

        Args:
            labels (list of str): Labels for each variable.
        """
        self.set_thetagrids(angles=np.degrees(self.theta), labels=labels)
        self.tick_params(axis='x', labelsize=label_fontsize)

    def _gen_axes_patch(self):
        """
        Generate the base patch for the radar axes based on the frame type.

        Returns:
            matplotlib.patches.Patch: Patch object for the radar axes.
        """
        if self.frame == "circle":
            return Circle((0.5, 0.5), 0.5)
        if self.frame == "polygon":
            return RegularPolygon((0.5, 0.5), self.num_vars, radius=0.5, edgecolor="k")

        raise ValueError(f"unknown value for 'frame': {self.frame}")

    def draw(self, renderer):
        """
        Custom draw method. If frame is polygon, make gridlines polygon-shaped.
        """
        if self.frame == "polygon":
            gridlines = self.yaxis.get_gridlines()
            for gl in gridlines:
                gl.get_path()._interpolation_steps = self.num_vars
        super().draw(renderer)

    def _gen_axes_spines(self):
        """
        Generate the spines for the radar axes based on the frame type.

        Returns:
            dict: A dictionary of spines for the radar axes.
        """
        if self.frame == "circle":
            return super()._gen_axes_spines()
        if self.frame == "polygon":
            spine = Spine(
                axes=self,
                spine_type="circle",
                path=Path.unit_regular_polygon(self.num_vars),
            )
            spine.set_transform(
                Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
            )
            return {"polar": spine}

        raise ValueError(f"unknown value for 'frame': {self.frame}")


class RadarChart:
    """
    Class for creating and manipulating radar charts, using the RadarAxes custom projection.

    Attributes:
        num_vars (int): Number of variables on the radar chart.
        frame (str): Shape of the radar chart's frame, either 'circle' or 'polygon'.
        theta (numpy.ndarray): Array of angles for the variables.
        fig (matplotlib.figure.Figure): Figure object for the radar chart.
        ax (RadarAxes): Axes object for the radar chart.
    """

    def __init__(
        self,
        num_vars: int,
        frame: str = "circle",
        label_positions: List[Tuple[float, float]] = None,
        legend_bbox_to_anchor: Tuple[float, float] = (1.2, 0.7),
        figure_size: Tuple[int, int] = (20, 20),
        legend_handle_length: float = 2.0,
    ) -> None:
        """
        Initialize the RadarChart object.

        Args:
            num_vars (int): Number of variables in the radar chart.
            frame (str): Shape of the radar chart's frame.
            label_positions (List[Tuple[float, float]]): List of tuples with x and y coordinates for label positions.
            legend_bbox_to_anchor (Tuple[float, float]): Tuple with x and y coordinates for legend positioning.
        """
        self.num_vars = num_vars
        self.frame = frame
        self.figure_size = figure_size
        self.label_positions = (
            label_positions if label_positions is not None else [(0, 0)] * num_vars
        )
        self.legend_bbox_to_anchor = legend_bbox_to_anchor
        self.legend_handle_length = legend_handle_length
        self.theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        self._register_radar_projection()  # Register custom projection

        self.fig = None
        self.ax = None

    @staticmethod
    def configure_plot_style() -> None:
        """Configure the plot style for radar charts."""
        sns.set(font_scale=1.5)
        plt.style.use("seaborn-v0_8-white")
        plt.rcParams["ytick.labelleft"] = True
        plt.rcParams["xtick.labelbottom"] = True

    def _register_radar_projection(self) -> None:
        """Register the RadarAxes projection with Matplotlib."""
        register_projection(RadarAxes)

    def plot(
        self,
        data: pd.DataFrame,
        include_legend: bool = True,
        legend_labels: List[str] = None,
        line_styles: List[str] = None,
        colors: List[str] = None,
        grid_label_fontsize: int = 10,
        ax: RadarAxes = None,
        legend_fontsize: int = 12,
        plot_title: str = None,
        title_pad: int = 10,
        label_fontsize: int = 12,
    ) -> None:
        """
        Plot the radar chart with the given DataFrame.

        Args:
            data (pd.DataFrame): DataFrame with columns as variables and rows as datasets.
            title (Optional[str]): Title of the chart.
            include_legend (bool): Flag to include the legend.
        """
        self.num_vars = len(data.columns)
        self.theta = np.linspace(0, 2 * np.pi, self.num_vars, endpoint=False)

        if colors is None:
            colors = [
                "#DC143C",  # Crimson
                "#FFA500",  # Orange
                "#4169E1",  # Royal Blue
                "#32CD32",  # Lime Green
                "#BA55D3",  # Medium Orchid
                "#00CED1",  # Cyan
            ]

        if line_styles is None:
            line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 10))]

        if legend_labels is None:
            legend_labels = list(data.index)

        RadarChart.configure_plot_style()

        if ax is None:
            self.fig, ax = plt.subplots(
                figsize=self.figure_size,
                subplot_kw=dict(
                    projection="radar", frame=self.frame, num_vars=self.num_vars
                ),
            )

        self.ax = ax

        if self.fig is not None:
            self.fig.subplots_adjust(top=0.85, bottom=0.05)

        for (label, d), (label_orig, d_orig), color, line_style in zip(
            data.iterrows(), data.iterrows(), colors, line_styles
        ):
            self.ax.plot(
                self.theta,
                d,
                label=label,
                color=color,
                linewidth=2.0,
                linestyle=line_style,
            )
            self.ax.fill(self.theta, d, alpha=0.25, color=color)

            # # Annotate each point with its value
            # for i, value in enumerate(d_orig):
            #     self.ax.text(
            #         self.theta[i],
            #         d[i],
            #         f"{value:.2f}",
            #         horizontalalignment="center",
            #         verticalalignment="center",
            #         fontsize=10,
            #         color=color,
            #         weight="bold",
            #         bbox=dict(facecolor="white", edgecolor="none", pad=1),
            #     )

        # Set variable labels and adjust specific label positions
        self.ax.set_varlabels(data.columns, label_fontsize=label_fontsize)
        self._adjust_label_positions()

        # Set radial grid range from 0 to 1
        self.ax.set_rgrids(
            np.arange(0, 1.1, 0.2),
            labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            fontsize=grid_label_fontsize,
        )

        # Add legend if required
        if include_legend:
            custom_handles = [
                plt.Line2D([0], [0], color=color, lw=2, linestyle=ls)
                for color, ls in zip(
                    colors[: len(legend_labels)], line_styles[: len(legend_labels)]
                )
            ]

            self.ax.legend(
                handles=custom_handles,
                labels=legend_labels,
                loc="upper right",
                bbox_to_anchor=self.legend_bbox_to_anchor,
                handlelength=self.legend_handle_length,
                prop={'size': legend_fontsize, 'weight': 'bold'},
            )

        # Set the title for the subplot
        if plot_title is not None:
            self.ax.set_title(plot_title, fontsize=legend_fontsize, fontweight='bold', pad=title_pad)

        plt.tight_layout()

    def _adjust_label_positions(self) -> None:
        """Adjust positions of specific labels on the radar chart."""
        labels = self.ax.get_xticklabels()
        for label, (x, y) in zip(labels, self.label_positions):
            label.set_position((x, y))

    def save_figure(self, filename: str, format: str = "pdf") -> None:
        """
        Save the radar chart to a file.

        Args:
            filename (str): Name of the file to save the chart.
            format (str): Format of the saved file.
        """
        self.fig.savefig(f"{filename}.{format}", format=format)
