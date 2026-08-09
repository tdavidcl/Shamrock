"""
CI test: sink state stays synchronized across MPI ranks through dump/reload
===========================================================================

Creates a tiny SPH setup with a few sinks and gas particles, checks sink sync,
evolves, dumps, reloads into a fresh context, and checks again.
"""

import shamrock

DUMP_NAME = "sink_sync_test.sham"


def check_sinks_are_in_sync(ctx, model):
    s = str(model.get_sinks())
    # Collective: every rank must call this
    hist = shamrock.algs.all_string_histogram([s], delimiter="\n", hash_based=False)
    if len(hist) != 1:
        raise RuntimeError(f"sinks not in sync across ranks: {hist}")
    key, count = next(iter(hist.items()))
    if count != shamrock.sys.world_size():
        raise RuntimeError(
            f"expected count={shamrock.sys.world_size()}, got {count} for key={key!r}"
        )
    shamrock.sys.mpi_barrier()
    if shamrock.sys.world_rank() == 0:
        print("Sinks are in sync !")


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


def build_model_with_sinks():
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    cfg.set_eos_isothermal(1.0)
    cfg.set_particle_mass(1e-3)
    cfg.set_boundary_periodic()
    cfg.set_show_cfl_detail(True)
    cfg.set_units(codeu)
    model.set_solver_config(cfg)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)
    model.set_eta_sink(1.0)

    model.init_scheduler(100, 1)

    # Very coarse HCP cube -> handful of SPH particles
    dr = 0.1
    bmin = (-0.6, -0.6, -0.6)
    bmax = (0.6, 0.6, 0.6)
    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen)

    # A few sinks (must be added after init_scheduler, on all ranks)
    model.add_sink(1.0, (0.1, 0.0, 0.0), (0.0, 0.05, 0.0), 0.05)
    model.add_sink(0.5, (-0.2, 0.1, 0.0), (0.0, -0.03, 0.0), 0.04)
    model.add_sink(0.25, (0.0, -0.15, 0.05), (0.02, 0.0, 0.0), 0.03)

    return ctx, model


def main():
    ctx, model = build_model_with_sinks()

    check_sinks_are_in_sync(ctx, model)

    for _ in range(3):
        model.timestep()
    check_sinks_are_in_sync(ctx, model)

    sinks_before_dump = str(model.get_sinks())
    model.dump(DUMP_NAME)

    del model
    del ctx

    ctx2 = shamrock.Context()
    ctx2.pdata_layout_new()
    model2 = shamrock.get_Model_SPH(context=ctx2, vector_type="f64_3", sph_kernel="M4")
    model2.load_from_dump(DUMP_NAME)

    sinks_after_reload = str(model2.get_sinks())
    if sinks_before_dump != sinks_after_reload:
        raise RuntimeError(
            "sink content changed across dump/reload:\n"
            f"  before={sinks_before_dump!r}\n"
            f"  after ={sinks_after_reload!r}"
        )

    check_sinks_are_in_sync(ctx2, model2)

    for _ in range(3):
        model2.timestep()
    check_sinks_are_in_sync(ctx2, model2)

    if shamrock.sys.world_rank() == 0:
        print("run_test_sink_synchro: OK")


main()
