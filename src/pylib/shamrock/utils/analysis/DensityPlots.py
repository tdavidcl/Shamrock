import numpy as np

from .StandardPlotHelper import StandardPlotHelper


def get_rhod_getter(model, jdust, ndust):

    def int_getter(size: int, dic_out: dict) -> np.array:
        s_j = dic_out["s_j"].reshape(-1, ndust)
        return s_j[:, jdust] ** 2

    return int_getter


def get_epsilon_getter(model, jdust, ndust):

    pmass = model.get_particle_mass()
    hfact = model.get_hfact()

    rhod_getter = get_rhod_getter(model, jdust, ndust)

    def int_getter(size: int, dic_out: dict) -> np.array:

        rhod = rhod_getter(size, dic_out)

        hpart = dic_out["hpart"]
        rho = pmass * (hfact / np.array(hpart)) ** 3

        return rhod / rho

    return int_getter


def ColumnDensityPlot(model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix):
    def compute_rho_integ(helper):
        return helper.column_integ_render("rho", "f64")

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rho_integ,
    )


def ColumnDensityPlotDust(
    model, ext_r, nx, ny, ex, ey, center, analysis_folder, analysis_prefix, jdust, ndust
):
    def compute_rhod_integ(helper):
        return helper.column_integ_render(
            "custom", "f64", custom_getter=get_rhod_getter(model, jdust, ndust)
        )

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rhod_integ,
    )


def SliceDensityPlot(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_slice(helper):
        return helper.slice_render("rho", "f64", do_normalization, min_normalization)

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rho_slice,
    )


def SliceDensityPlotDust(
    model,
    ext_r,
    nx,
    ny,
    ex,
    ey,
    center,
    analysis_folder,
    analysis_prefix,
    jdust,
    ndust,
    do_normalization=True,
    min_normalization=1e-9,
):
    def compute_rho_slice(helper):
        return helper.slice_render(
            "custom",
            "f64",
            do_normalization,
            min_normalization,
            custom_getter=get_rhod_getter(model, jdust, ndust),
        )

    return StandardPlotHelper(
        model,
        ext_r,
        nx,
        ny,
        ex,
        ey,
        center,
        analysis_folder,
        analysis_prefix,
        compute_function=compute_rho_slice,
    )
