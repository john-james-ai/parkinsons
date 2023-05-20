#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /parkinsons/analysis/task.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 3rd 2023 11:06:40 pm                                                  #
# Modified   : Saturday May 20th 2023 03:41:25 pm                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module for analysis of the Subject Dataset"""
import seaborn as sns

from parkinsons.analysis.base import Dataset
from parkinsons.config import Visual

sns.set_style(Visual.style)
sns.set_palette = sns.dark_palette(
    Visual.palette.color, reverse=Visual.palette.reverse, as_cmap=Visual.palette.as_cmap
)
# ------------------------------------------------------------------------------------------------ #


class Task(Dataset):
    def __init__(self, name: str, filepath: str) -> None:
        super().__init__(name=name, filepath=filepath)
        self._df["Duration"] = self._df["End"] - self._df["Begin"]
