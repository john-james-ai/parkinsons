#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /parkinsons/analysis/subject.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 3rd 2023 11:06:40 pm                                                  #
# Modified   : Friday May 19th 2023 04:23:49 am                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module for analysis of the Subject Dataset"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from parkinsons.analysis.base import Dataset, Canvas
from parkinsons.config import Visual

sns.set_style(Visual.style)
sns.set_palette = sns.dark_palette(
    Visual.palette.color, reverse=Visual.palette.reverse, as_cmap=Visual.palette.as_cmap
)
# ------------------------------------------------------------------------------------------------ #


class Subject(Dataset):
    def __init__(self, name: str, filepath: str) -> None:
        super().__init__(name=name, filepath=filepath)
        self._updrs = pd.melt(
            self._df[["Subject", "Visit", "Sex", "YearsSinceDx", "UPDRSIII_Off", "UPDRSIII_On"]],
            id_vars=["Subject", "Visit", "Sex", "YearsSinceDx"],
            value_vars=["UPDRSIII_Off", "UPDRSIII_On"],
            var_name="Instrument",
            value_name="Score",
        )

    def histogram(
        self, x: str, grouping: str = None, title: str = None, ax: plt.axes = None
    ) -> None:
        """Plots the distribution of age by sex.

        Args:
            ax (plt.axes): A matplotlib axes object
        """
        ax = ax or Canvas().ax

        sns.histplot(
            data=self._df, x=x, hue=grouping, multiple="stack", ax=ax, palette="Blues_r"
        ).set(title=title)

    def updrs_gender(self, ax: plt.axes = None) -> pd.DataFrame:
        """Prints the distribution of UPDRS scores by gender, on and off medication.

        Args:
            ax (plt.axes): An matplotlib axes object.
        """
        suptitle = "Unified Parkinson's Disease Rating Scale (UPDRS)"
        title_male = "Male Subjects"
        title_female = "Female Subjects"
        nrows = 1
        ncols = 2
        canvas = Canvas(nrows=nrows, ncols=ncols)
        fig = canvas.fig
        axs = canvas.axs
        fig.suptitle(suptitle)

        # Melt the dataframe by instrument, then extract, male and female datasets

        male_dataset = self._updrs[self._updrs["Sex"] == "M"]
        female_dataset = self._updrs[self._updrs["Sex"] == "F"]
        sns.histplot(data=female_dataset, x="Score", hue="Instrument", kde=True, ax=axs[0]).set(
            title=title_female
        )
        sns.histplot(data=male_dataset, x="Score", hue="Instrument", kde=True, ax=axs[1]).set(
            title=title_male
        )
        fig.tight_layout()

    def updrs_med(self, ax: plt.axes = None) -> pd.DataFrame:
        """Prints the distribution of UPDRS scores by medication condition and gender.

        Args:
            ax (plt.axes): An matplotlib axes object.
        """
        suptitle = "Unified Parkinson's Disease Rating Scale (UPDRS)"
        title_off = "Off Medication"
        title_on = "On Medication"
        nrows = 1
        ncols = 2
        canvas = Canvas(nrows=nrows, ncols=ncols)
        fig = canvas.fig
        axs = canvas.axs
        fig.suptitle(suptitle)

        updrs_on = self._updrs[self._updrs["Instrument"] == "UPDRSIII_On"]
        updrs_off = self._updrs[self._updrs["Instrument"] == "UPDRSIII_Off"]
        sns.histplot(data=updrs_off, x="Score", hue="Sex", kde=True, ax=axs[0]).set(title=title_off)
        sns.histplot(data=updrs_on, x="Score", hue="Sex", kde=True, ax=axs[1]).set(title=title_on)
        fig.tight_layout()

    def describe_updrs(
        self,
        column: str = None,
        filter_var: str = None,
        filter_val: str = None,
        group_var: str = None,
    ) -> pd.DataFrame:
        return (
            self._updrs.loc[self._updrs[filter_var] == filter_val]
            .groupby(by=group_var)[column]
            .describe()
        )
