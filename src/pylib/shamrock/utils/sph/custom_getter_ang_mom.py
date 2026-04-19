import numpy as np
import shamrock.sys

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def gen_angular_mom_custom_getter(model, Lprojection=[0.0, 0.0, 1.0]):

    if _HAS_NUMBA:
        if shamrock.sys.world_rank() == 0:
            print("Using numba for angular momentum in gen_angular_mom_custom_getter")

    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    def internal(
        size: int,
        hpart: np.array,
        x: np.array,
        y: np.array,
        z: np.array,
        vx: np.array,
        vy: np.array,
        vz: np.array,
    ) -> np.array:

        rho = pmass * (hfact / hpart) ** 3

        r = np.stack([x, y, z], axis=-1)
        v = np.stack([vx, vy, vz], axis=-1)
        L = rho[:, None] * np.cross(r, v)

        # project L onto the Lprojection vector
        L = np.sum(L * Lprojection, axis=-1)

        return L

    if _HAS_NUMBA:
        internal = njit(internal)

    def custom_getter(size: int, dic_out: dict) -> np.array:
        return internal(
            size,
            dic_out["hpart"],
            dic_out["xyz"][:, 0],
            dic_out["xyz"][:, 1],
            dic_out["xyz"][:, 2],
            dic_out["vxyz"][:, 0],
            dic_out["vxyz"][:, 1],
            dic_out["vxyz"][:, 2],
        )

    return custom_getter
