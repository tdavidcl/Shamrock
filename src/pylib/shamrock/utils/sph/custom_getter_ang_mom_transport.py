import numpy as np
import shamrock.sys

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def gen_angular_mom_transport_custom_getter(model, velocity_profile):
    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    if _HAS_NUMBA:
        if shamrock.sys.world_rank() == 0:
            print("Using numba for velocity profile in gen_angular_mom_transport_custom_getter")
        vel_profile_jit = njit(velocity_profile)
    else:
        vel_profile_jit = np.vectorize(velocity_profile)

    def internal(
        x: np.array,
        y: np.array,
        z: np.array,
        vx: np.array,
        vy: np.array,
        vz: np.array,
        hpart: np.array,
        cs: np.array,
    ) -> np.array:
        rho = pmass * (hfact / hpart) ** 3
        P = cs**2 * rho  # TODO: use true pressure

        r = np.sqrt(x**2 + y**2)
        r_safe = r + 1e-9
        v_r = (x * vx + y * vy) / r_safe
        v_theta = (-y * vx + x * vy) / r_safe

        delta_vtheta = v_theta - vel_profile_jit(r)
        alpha = rho * v_r * delta_vtheta / P

        return alpha

    if _HAS_NUMBA:
        if shamrock.sys.world_rank() == 0:
            print("Using numba for custom getter in gen_angular_mom_transport_custom_getter")
        internal = njit(internal)

    def custom_getter(size: int, dic_out: dict) -> np.array:
        return internal(
            dic_out["xyz"][:, 0],
            dic_out["xyz"][:, 1],
            dic_out["xyz"][:, 2],
            dic_out["vxyz"][:, 0],
            dic_out["vxyz"][:, 1],
            dic_out["vxyz"][:, 2],
            dic_out["hpart"],
            dic_out["soundspeed"],
        )

    return custom_getter
