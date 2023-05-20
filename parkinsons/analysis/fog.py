#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /parkinsons/analysis/fog.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 3rd 2023 06:10:11 pm                                                  #
# Modified   : Wednesday May 3rd 2023 11:08:44 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from parkinsons.analysis.base import Dataset
from parkinsons.config import Visual

sns.set_style(Visual.style)
sns.set_palette = sns.dark_palette(
    Visual.palette.color, reverse=Visual.palette.reverse, as_cmap=Visual.palette.as_cmap
)
# ------------------------------------------------------------------------------------------------ #


class FOGDataset(Dataset):
    """Provides data and visualizations for a Fog Dataset.

    Args:
        filepath (str): Path to metadata file
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._df = pd.read_csv(self._filepath)
        self._info = None
        self._subject_summary = None

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._flog.head(n)

    def subject_summary(self) -> pd.DataFrame:
        """Provides a summary of series, visits, tests, and medication by subject"""
        fig, axs = plt.subplots(2, 2, figsize=Visual.figsize)
        fig.suptitle("tDCS Fog Metadata Summary")
        self.subject_series(axs[0, 0])
        self.subject_visits(axs[0, 1])
        self.subject_tests(axs[1, 0])
        self.subject_meds(axs[1, 1])
        fig.tight_layout()
        counts = self._df.groupby(by="Subject").nunique()
        return counts.describe().T

    def subject_series(self, ax: plt.axes) -> None:
        """Plots subjects b the number of series."""
        d = self._df.groupby(by="Subject")["Id"].nunique().value_counts().to_frame().reset_index()
        d.columns = ["Series", "Subjects"]
        sns.barplot(data=d, x="Series", y="Subjects", palette="Blues_r", ax=ax).set(
            title="Subjects by Num Series"
        )
        for container in ax.containers:
            ax.bar_label(container)
        return ax

    def subject_visits(self, ax: plt.axes) -> None:
        """Plots the number of visits for subjects."""
        d = (
            self._df.groupby(by="Subject")["Visit"]
            .nunique()
            .value_counts()
            .to_frame()
            .reset_index()
        )
        d.columns = ["Visits", "Subjects"]
        sns.barplot(data=d, x="Visits", y="Subjects", palette="Blues_r", ax=ax).set(
            title="Subjects by Number of Visits"
        )
        for container in ax.containers:
            ax.bar_label(container)
        return ax

    def subject_tests(self, ax: plt.axes) -> None:
        """Plots the number of visits for subjects."""
        d = self._df.groupby(by="Subject")["Test"].nunique().value_counts().to_frame().reset_index()
        d.columns = ["Tests", "Subjects"]
        sns.barplot(data=d, x="Tests", y="Subjects", palette="Blues_r", ax=ax).set(
            title="Subjects by Number of Tests"
        )
        for container in ax.containers:
            ax.bar_label(container)
        return ax

    def subject_meds(self, ax: plt.axes) -> None:
        """Plots the number of visits for subjects."""
        d = (
            self._df.groupby(by="Subject")["Medication"]
            .nunique()
            .value_counts()
            .to_frame()
            .reset_index()
        )
        d.columns = ["Medication Condition", "Subjects"]
        sns.barplot(data=d, x="Medication Condition", y="Subjects", palette="Blues_r", ax=ax).set(
            title="Subjects by Medication Condition"
        )
        for container in ax.containers:
            ax.bar_label(container)
        return ax
