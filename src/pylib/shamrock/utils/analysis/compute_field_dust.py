import numpy as np

from shamrock.utils.numba_helper import maybe_njit



def compute_s_mean_field(model):
    codeu = model.get_units()

    cfg_json = model.get_current_config().to_json()
    drag_mode = cfg_json["dust_config"]["drag_mode"]

    ndust = cfg_json["dust_config"]["mode"]["ndust"]
    grain_size = drag_mode["grains_sizes"]

    def int_getter(size: int, dic_out: dict,
        ndust: int = ndust,
        grain_size: np.ndarray = np.asarray(grain_size),) -> np.array:
        s_j = dic_out["s_j"].reshape(-1, ndust)

        rho_d = s_j**2

        rho_d_integ = np.sum(rho_d, axis=1)
        rho_d_s_integ = np.sum(rho_d * grain_size, axis=1)

        s_mean = rho_d_s_integ / rho_d_integ
        return s_mean

    return model.compute_field("custom", "f64",maybe_njit(int_getter))


def compute_dlog_s_mean_dt_field(model):
    codeu = model.get_units()

    cfg_json = model.get_current_config().to_json()
    drag_mode = cfg_json["dust_config"]["drag_mode"]

    ndust = cfg_json["dust_config"]["mode"]["ndust"]
    grain_size = drag_mode["grains_sizes"]

    def int_getter(size: int, dic_out: dict,
        ndust: int = ndust,
        grain_size: np.ndarray = np.asarray(grain_size),) -> np.array:
        s_j = dic_out["s_j"].reshape(-1, ndust)
        ds_j_dt = dic_out["ds_j_dt"].reshape(-1, ndust)

        rho_d = s_j**2
        drhod_dt = 2 * s_j * ds_j_dt

        rho_d_integ = np.sum(rho_d, axis=1)
        drhod_dt_integ = np.sum(drhod_dt, axis=1)

        rho_d_s_integ = np.sum(rho_d * grain_size, axis=1)
        drho_d_s_dt_integ = np.sum(drhod_dt * grain_size, axis=1)

        s_mean = rho_d_s_integ / rho_d_integ
        ds_mean_dt = (
            drho_d_s_dt_integ * rho_d_integ - drhod_dt_integ * rho_d_s_integ
        ) / rho_d_integ**2

        return ds_mean_dt / s_mean

    return model.compute_field("custom", "f64",maybe_njit(int_getter))
