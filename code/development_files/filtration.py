# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:07:22 2024

@author: DanielEspinoza
"""

import numpy as np
import numpy.typing as npt


def simulate_df(parameters: dict, inlet_profile: npt.ArrayLike) -> np.ndarray:
    raise NotImplementedError


def simulate_iex(parameters: dict, inlet_profile: npt.ArrayLike) -> np.ndarray:
    """
    Simulate Ion Exchange Chromatography using CADET-Process.

    This is Joaquin's model.

    Parameters
    ----------
    parameters : dict
        DESCRIPTION.
    inlet_profile : npt.ArrayLike
        DESCRIPTION.

    Returns
    -------
    np.ndarray
        The outlet profile of the ion exchange unit operation.

    """
    raise NotImplementedError


def simulate_ufdf(parameters: dict, inlet_profile: npt.ArrayLike) -> np.ndarray:
    raise NotImplementedError


def simulate_system(parameters: dict, inlet_profile: npt.ArrayLike) -> np.ndarray:
    inlet_profile = parameters["inlet_profile"]

    def_parameters = parameters["def_parameters"]
    outlet_def = simulate_df(def_parameters, inlet_profile)

    iex_parameters = parameters["iex_parameters"]
    outlet_iex = simulate_iex(iex_parameters, outlet_def)

    ufdf_parameters = parameters["ufdf_parameters"]
    outlet_ufdf = simulate_ufdf(ufdf_parameters, outlet_iex)

    return outlet_ufdf
