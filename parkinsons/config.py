#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /parkinsons/config.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 3rd 2023 04:37:25 pm                                                  #
# Modified   : Thursday May 4th 2023 04:52:06 pm                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass


@dataclass
class Palette:
    color: str = "#69d"
    reverse: bool = True
    as_cmap: bool = True


# ------------------------------------------------------------------------------------------------ #
class Visual:
    style = "whitegrid"
    palette = Palette
    figsize = (12, 3)
