#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /parkinsons/analysis/base.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 3rd 2023 09:52:47 pm                                                  #
# Modified   : Saturday May 20th 2023 04:02:01 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC
from dataclasses import dataclass, field
import logging
from typing import List
import numpy as np
import scipy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from parkinsons.config import Visual

sns.set_style(Visual.style)
sns.set_palette = sns.dark_palette(
    Visual.palette.color, reverse=Visual.palette.reverse, as_cmap=Visual.palette.as_cmap
)

# ------------------------------------------------------------------------------------------------ #
PERCENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# ------------------------------------------------------------------------------------------------ #
#                                            CANVAS                                                #
# ------------------------------------------------------------------------------------------------ #


@dataclass
class Canvas:
    nrows: int = 1
    ncols: int = 1
    fig: plt.figure = None
    ax: plt.axes = None
    axs: List = field(default_factory=lambda: [plt.axes])

    def __post_init__(self) -> None:
        if self.nrows > 1 or self.ncols > 1:
            figsize = []
            figsize.append(Visual.figsize[0])
            figsize.append(Visual.figsize[1] * self.nrows)
            self.fig, self.axs = plt.subplots(self.nrows, self.ncols, figsize=figsize)
        else:
            self.fig, self.ax = plt.subplots(self.nrows, self.ncols, figsize=Visual.figsize)


# ------------------------------------------------------------------------------------------------ #
#                                            DATASET                                               #
# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    """Provides data and visualizations for a Fog Dataset.

    Args:
        filepath (str): Path to metadata file
    """

    def __init__(self, name: str, filepath: str) -> None:
        self._name = name
        self._filepath = filepath
        self._df = pd.read_csv(self._filepath)
        self._info = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def reset(self) -> None:
        self._info = None
        self._summary = None

    def info(self, table: bool = False, plot: bool = True) -> pd.DataFrame:
        """Returns a DataFrame with basic dataset statistics"""

        self._info_table()

        if plot:
            self._info_plot()
        if table:
            return round(self._info, 2)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n)

    def sample(
        self, n: int = 5, frac: float = None, replace: bool = False, random_state: int = None
    ) -> pd.DataFrame:
        """Returns a sample from the FOG Dataset

        Args:
            n (int): Number of items to return. Defaults to five.
            frac (float): Proportion of items to return
            replace (bool): Whether to sample with replacement
            random_state (int): Pseudo random seed.
        """
        return self._df.sample(n=n, frac=frac, replace=replace, random_state=random_state)

    def get(self, x: str = None, filter_var: str = None, filter_val: str = None) -> pd.DataFrame():
        """Subsets a dataset based upon a single filter variable and value

        Args:
            x (str): The name of the variable to select.
            filter_var (str): Filter variable
            filter_val (str): Filter value

        """
        df = self._df
        if x is not None:
            df = self._df[x]
        if filter_var:
            return df[df[filter_var] == filter_val]
        return df

    def compare(self, a: np.array, b: np.array) -> pd.DataFrame:
        """Compares two distributions using Anderson-Darling k-sample test.

        Args:
            a (Union[np.array,list]): Numpy array or list like.
            b (Union[np.array,list]): Numpy array or list like.
        """
        return scipy.stats.anderson_ksamp([a, b])

    def _info_table(self) -> pd.DataFrame:
        """Prints Dataset info in table format."""

        if self._info is None:
            self._info = self._df.dtypes.to_frame().reset_index()
            self._info.columns = ["Column", "Dtype"]
            self._info["Valid"] = self._df.count().values
            self._info["Null"] = self._df.isna().sum().values
            self._info["Validity"] = self._info["Valid"] / self._df.shape[0]
            self._info["Unique"] = self._df.nunique().values
            self._info["Cardinality"] = self._info["Unique"] / self._df.shape[0]
            self._info["Size"] = (
                self._df.memory_usage(deep=True, index=False).to_frame().reset_index()[0]
            )
            self._info = round(self._info, 2)

    def _info_plot(self) -> None:
        """Plots the number or proportion of valid values by column

        Args:
            normalize (bool): If True, use Validity column; otherwise, use Valid.
        """
        ncols = 2
        nrows = 2
        canvas = Canvas(ncols=ncols, nrows=nrows)
        fig = canvas.fig
        axs = canvas.axs

        fig.suptitle(f"{self._name} Dataset Summary")

        self.plot_cardinality(axs[0, 0])
        self.plot_cardinality(axs[0, 1], normalized=True)
        self.plot_validity(axs[1, 0])
        self.plot_validity(axs[1, 1], normalized=True)
        fig.tight_layout()

    def plot_dtypes(self, ax: plt.axes) -> None:
        """Plots counts by dtype."""
        data = self._info.groupby(by="Dtype")["Valid"].sum().to_frame().reset_index()
        sns.barplot(data=data, x="Dtype", y="Valid", palette="Blues_r", ax=ax).set(
            title="Counts by Dtype",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        for container in ax.containers:
            ax.bar_label(container)
        return ax

    def plot_validity(self, ax: plt.axes, normalized: bool = False) -> None:
        """Plots validity by column"""
        if normalized:
            data = self._info[["Column", "Validity"]]
            sns.barplot(data=data, x="Column", y="Validity", palette="Blues_r", ax=ax).set(
                title="Validity"
            )
        else:
            data = self._info[["Column", "Valid"]]
            sns.barplot(data=data, x="Column", y="Valid", palette="Blues_r", ax=ax).set(
                title="Validity"
            )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        for container in ax.containers:
            ax.bar_label(container)
        return ax

    def plot_cardinality(self, ax: plt.axes, normalized: bool = False) -> None:
        """Plots cardinality by column."""
        if normalized:
            data = self._info[["Column", "Cardinality"]]
            sns.barplot(data=data, x="Column", y="Cardinality", palette="Blues_r", ax=ax).set(
                title="Cardinality"
            )
        else:
            data = self._info[["Column", "Unique"]]
            sns.barplot(data=data, x="Column", y="Unique", palette="Blues_r", ax=ax).set(
                title="Cardinality"
            )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        for container in ax.containers:
            ax.bar_label(container)
        return ax

    def plot_size(self, ax: plt.axes) -> None:
        """Plots column sizes."""
        data = self._info[["Column", "Size"]]
        sns.barplot(data=data, x="Column", y="Size", palette="Blues_r", ax=ax).set(title="Size")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        for container in ax.containers:
            ax.bar_label(container)
        return ax

    def countplot(
        self,
        x: str,
        y: str = None,
        hue: str = None,
        title: str = None,
        orient: str = "v",
        ax: plt.axes = None,
    ) -> plt.axes:
        """Produces plot of counts of observations in each categorical bin using bars.

        Args:
            x (str): The name of the variable to be plotted along the x axis.
            y (str): The name of the variable to be plotted along the y axis.
            hue (str): The name of the grouping variable.
            title (str): The title for the plot.
            orient (str): The vertical or horizontal orientation of the plot.
            ax (plt.axes): A matplotlib axes object.

        """
        ax = ax or Canvas().ax
        title = title or x
        sns.countplot(
            data=self._df, x=x, y=y, hue=hue, orient=orient, palette="Blues_r", ax=ax
        ).set(title=title)
        ax.bar_label(ax.containers[0], label_type="edge")
        return ax

    def histplot(
        self, x: str, y: str = None, hue: str = None, title: str = None, ax: plt.axes = None
    ) -> plt.axes:
        """Produces univariate and bivariate histograms to show distributions of datasets.

        Args:
            x (str): The name of the variable to be plotted along the x axis.
            y (str): The name of the variable to be plotted along the y axis.
            hue (str): The name of the grouping variable.
            title (str): A title for the plot.
            ax (plt.axes): A matplotlib axes object.

        """
        if title is None:
            title = x if hue is None else x + " by " + hue
        ax = ax or Canvas().ax
        sns.histplot(data=self._df, x=x, y=y, hue=hue, palette="Blues_r", kde=True, ax=ax).set(
            title=title
        )
        return ax

    def boxplot(
        self, x: str, y: str = None, hue: str = None, title: str = None, ax: plt.axes = None
    ) -> plt.axes:
        """Produces univariate and bivariate histograms to show distributions of datasets.

        Args:
            x (str): The name of the variable to be plotted along the x axis.
            y (str): The name of the variable to be plotted along the y axis.
            hue (str): The name of the grouping variable.
            title (str): A title for the plot.
            ax (plt.axes): A matplotlib axes object.

        """
        if title is None:
            title = x if hue is None else x + " by " + hue
        ax = ax or Canvas().ax
        sns.boxplot(data=self._df, x=x, y=y, hue=hue, palette="Blues_r", ax=ax).set(title=title)
        return ax

    def barplot(
        self,
        x: str,
        y: str = None,
        hue: str = None,
        errorbar: str = None,
        title: str = None,
        ax: plt.axes = None,
    ) -> plt.axes:
        """Produces barplot.

        Args:
            x (str): The name of the variable to be plotted along the x axis.
            y (str): The name of the variable to be plotted along the y axis.
            hue (str): The name of the grouping variable.
            errorbar (str): Name of variable containing error bar information.
            title (str): A title for the plot.
            ax (plt.axes): A matplotlib axes object.

        """
        if title is None:
            title = x if hue is None else x + " by " + hue
        ax = ax or Canvas().ax
        sns.barplot(
            data=self._df, x=x, y=y, hue=hue, palette="Blues_r", ax=ax, errorbar=errorbar
        ).set(title=title)
        return ax

    def describe(self, column: str = None, verbose: bool = True) -> pd.DataFrame:
        """Produces descriptive statistics

        Args:
            column (str): Optional column upon which descriptive statistics will be computed.
            verbose (bool): Optional. Whether to produce full (verbose) descriptive statistics, or abbreviated.

        """
        if column is None:
            description = self._df.describe(percentiles=PERCENTILES).T
        else:
            description = self._df[column].describe(percentiles=PERCENTILES).to_frame().T

        if verbose:
            return description
        else:
            desc = round(description[["mean", "std"]], 1)
            desc["Average"] = (
                desc["mean"].astype(str) + " " + "\u00B1" + " " + desc["std"].astype(str)
            )
            desc = desc["Average"].to_frame()
            return desc

    def describe_group(
        self, group_by: str, column: str = None, verbose: bool = True
    ) -> pd.DataFrame:
        if column is None:
            description = self._df.groupby(by=group_by).describe(percentiles=PERCENTILES).T
        else:
            description = self._df.groupby(by=group_by)[column].describe(percentiles=PERCENTILES)

        if verbose:
            return description
        else:
            desc = round(description[["mean", "std"]], 1)
            desc["Average"] = (
                desc["mean"].astype(str) + " " + "\u00B1" + " " + desc["std"].astype(str)
            )
            desc = desc["Average"].to_frame()
            return desc
