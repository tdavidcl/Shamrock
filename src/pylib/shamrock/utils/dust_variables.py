"""
Helpers for the evolved dust variables of the SPH monofluid TVA solver.

The solver evolves, for each dust species :math:`j`, one of the following variables
(selected with ``cfg.set_dust_mode_monofluid_tva(..., dust_variable=...)``):

* ``"sqrt_rho_eps"`` : :math:`S_j = \\sqrt{\\rho \\epsilon_j}` (Hutchison et al. 2018), field ``s_j``
* ``"eps"`` : :math:`\\epsilon_j` (Price & Laibe 2015), field ``eps_j``
* ``"sqrt_eps_over_1m_eps"`` : :math:`s_j = \\sqrt{\\epsilon_j / (1 - \\epsilon_j)}`
  (Ballabio et al. 2018), field ``sb_j``

The maps below mirror ``shammodels/sph/math/dust_variables.hpp``.
"""

import os

import numpy as np

#: All the dust variables, in the order used for plots
ALL_DUST_VARIABLES = ["sqrt_rho_eps", "sqrt_eps_over_1m_eps", "eps"]

_FIELD_NAMES = {
    "sqrt_rho_eps": "s_j",
    "eps": "eps_j",
    "sqrt_eps_over_1m_eps": "sb_j",
}

#: Plot labels of the dust variables
LABELS = {
    "sqrt_rho_eps": r"$S_j=\sqrt{\rho\epsilon_j}$ (HPL18)",
    "eps": r"$\epsilon_j$ + hard limiter (PL15)",
    "sqrt_eps_over_1m_eps": r"$s_j=\sqrt{\epsilon_j/(1-\epsilon_j)}$ (Ballabio+18)",
}

#: Plot colors of the dust variables (one fixed color per variable across every figure)
COLORS = {
    "sqrt_rho_eps": "tab:blue",
    "eps": "tab:orange",
    "sqrt_eps_over_1m_eps": "tab:green",
}

#: Plot markers of the dust variables
MARKERS = {
    "sqrt_rho_eps": "o",
    "eps": "s",
    "sqrt_eps_over_1m_eps": "^",
}


def field_name(dust_variable):
    """Name of the patch field holding the evolved dust variable."""
    return _FIELD_NAMES[dust_variable]


def deriv_field_name(dust_variable):
    """Name of the patch field holding the time derivative of the evolved dust variable."""
    return "d" + _FIELD_NAMES[dust_variable] + "_dt"


def eps_to_var(dust_variable, eps, rho):
    """Evolved dust variable from the dust fraction ``eps`` and the total density ``rho``."""
    if dust_variable == "sqrt_rho_eps":
        return np.sqrt(rho * eps)
    if dust_variable == "eps":
        return eps
    if dust_variable == "sqrt_eps_over_1m_eps":
        return np.sqrt(eps / (1 - eps))
    raise ValueError(f"unknown dust variable {dust_variable}")


def var_to_eps(dust_variable, X, rho):
    """Dust fraction from the evolved dust variable ``X`` and the total density ``rho``."""
    if dust_variable == "sqrt_rho_eps":
        return X**2 / rho
    if dust_variable == "eps":
        return X
    if dust_variable == "sqrt_eps_over_1m_eps":
        return X**2 / (1 + X**2)
    raise ValueError(f"unknown dust variable {dust_variable}")


def selected_dust_variables(hard_limiter=True):
    """
    Dust variables to run in a comparison.

    ``eps`` is not positivity preserving, so it is only included when the hard limiter
    (``ensure_s_j_positivity``, i.e. :math:`\\epsilon_j = \\max(\\epsilon_j, 0)`) is enabled.
    The list can be trimmed with the ``DUST_VARIABLES`` environment variable
    (comma separated, e.g. ``DUST_VARIABLES=sqrt_rho_eps,eps``).
    """
    variants = [v for v in ALL_DUST_VARIABLES if v != "eps" or hard_limiter]

    env = os.environ.get("DUST_VARIABLES")
    if env is not None:
        requested = [v.strip() for v in env.split(",") if v.strip() != ""]
        for v in requested:
            if v not in ALL_DUST_VARIABLES:
                raise ValueError(f"unknown dust variable {v} in DUST_VARIABLES")
        variants = [v for v in variants if v in requested]

    return variants
