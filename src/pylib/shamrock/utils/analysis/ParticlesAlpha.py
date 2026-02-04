from .StandardPlotHelper import StandardPlotHelper


def SliceAlphaAV(
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
    def compute_alpha_AV_slice(helper):
        return helper.slice_render("alpha_AV", "f64", do_normalization, min_normalization)

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
        compute_function=compute_alpha_AV_slice,
    )
