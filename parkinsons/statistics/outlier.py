#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /parkinsons/statistics/outlier.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday May 4th 2023 09:33:16 pm                                                   #
# Modified   : Thursday May 4th 2023 09:49:19 pm                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Union

import numpy as np


# ------------------------------------------------------------------------------------------------ #
class Outlier:
    __b = 1.4826
    __threshold = 3

    def univariate(self, a: Union[np.array, list]) -> np.array:
        """Univariate outlier detection and removal

        Args:
            a (Union[np.array,list]): An array or list like.
        """
        median = np.median(a)
        centered = np.abs(a - median)
        mad = np.median(centered) * self.__b
        return a[np.abs((a - median) / mad) < 3]
