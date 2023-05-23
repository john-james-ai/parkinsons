#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /parkinsons/data/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 3rd 2023 09:52:47 pm                                                  #
# Modified   : Saturday May 20th 2023 06:42:08 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from parkinsons.data.visual import Palette, Config, Canvas

sns.set_style(Config.style)
sns.set_palette = sns.dark_palette(Palette.blue, reverse=True, as_cmap=True)

# ------------------------------------------------------------------------------------------------ #
PERCENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


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

    def info(self) -> pd.DataFrame:
        """Returns a DataFrame with basic dataset statistics"""

        info = self._df.dtypes.to_frame().reset_index()
        info.columns = ["Column", "Dtype"]
        info["Valid"] = self._df.count().values
        info["Null"] = self._df.isna().sum().values
        info["Validity"] = info["Valid"] / self._df.shape[0]
        info["Unique"] = self._df.nunique().values
        info["Cardinality"] = info["Unique"] / self._df.shape[0]
        info["Size"] = self._df.memory_usage(deep=True, index=False).to_frame().reset_index()[0]
        info = round(info, 2)
        return info

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
            data=self._df, x=x, y=y, hue=hue, orient=orient, palette=Palette.blues_r, ax=ax
        ).set(title=title)
        ax.bar_label(ax.containers[0], label_type="edge")
        return ax

    def histplot(
        self,
        x: str,
        y: str = None,
        hue: str = None,
        multiple: str = None,
        title: str = None,
        ax: plt.axes = None,
    ) -> plt.axes:
        """Produces univariate and bivariate histograms to show distributions of datasets.

        Args:
            x (str): The name of the variable to be plotted along the x axis.
            y (str): The name of the variable to be plotted along the y axis.
            hue (str): The name of the grouping variable.
            multiple (str): Approach to resolving multiple elements when semantic mapping creates subsets. Only relevant with univariate data.
            title (str): A title for the plot.
            ax (plt.axes): A matplotlib axes object.

        """
        if title is None:
            title = x if hue is None else x + " by " + hue
        ax = ax or Canvas().ax
        sns.histplot(
            data=self._df,
            x=x,
            y=y,
            hue=hue,
            multiple=multiple,
            palette=Palette.blues_r,
            kde=True,
            ax=ax,
        ).set(title=title)
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
        sns.boxplot(data=self._df, x=x, y=y, hue=hue, palette=Palette.blues_r, ax=ax).set(
            title=title
        )
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
            data=self._df, x=x, y=y, hue=hue, palette=Palette.blues_r, ax=ax, errorbar=errorbar
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
