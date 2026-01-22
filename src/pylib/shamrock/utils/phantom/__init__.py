"""
Phantom related utilities.
"""

import os

import shamrock.sys


def parse_in_file(in_file):
    """
    Parse a Phantom .in file and return a dictionary of the parameters.
    """
    with open(in_file, "r") as f:
        lines = f.readlines()

    params = {}

    for line in lines:
        # Skip empty lines and comment lines
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # Check if line contains an equals sign
        if "=" in line:
            # Split by '=' to get variable name and value part
            parts = line.split("=", 1)
            var_name = parts[0].strip()

            # Get value part (everything after =)
            value_part = parts[1]

            # Remove comment if present (text after !)
            if "!" in value_part:
                value_part = value_part.split("!")[0]

            # Strip whitespace from value
            value = value_part.strip()

            # Try to convert to appropriate type
            # Check for boolean
            if value == "T":
                value = True
            elif value == "F":
                value = False
            else:
                # Try to convert to number
                try:
                    # Try integer first
                    if "." not in value and "E" not in value and "e" not in value:
                        value = int(value)
                    else:
                        # Try float
                        value = float(value)
                except ValueError:
                    # Keep as string if conversion fails
                    pass

            params[var_name] = value

    return params


def load_simulation(simulation_path, dump_file_name, in_file_name=None, do_print=True):
    """
    Load a Phantom simulation into a Shamrock model.
    """

    if do_print and shamrock.sys.world_rank() == 0:
        print("-----------------------------------------------------------")
        print("----------------   Phantom dump loading   -----------------")
        print("-----------------------------------------------------------")

    # setup = dump finish with .tmp
    is_setup_file = dump_file_name.endswith(".tmp")

    dump_path = os.path.join(simulation_path, dump_file_name)

    if in_file_name is not None:
        in_file_path = os.path.join(simulation_path, in_file_name)
        in_params = parse_in_file(in_file_path)
    else:
        in_params = None

    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Loading phantom dump from: ", dump_path)

    # Open the phantom dump
    dump = shamrock.load_phantom_dump(dump_path)

    # Start a SPH simulation from the phantom dump
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Generating Shamrock solver config from phantom dump")
    cfg = model.gen_config_from_phantom_dump(dump)
    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Setting Shamrock solver config")
    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)

    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Initializing domain scheduler")

    model.init_scheduler(int(1e8), 1)

    if do_print and shamrock.sys.world_rank() == 0:
        print(f" - Initializing from phantom dump (setup file: {is_setup_file})")

    if is_setup_file:
        model.init_from_phantom_dump(dump, 0.5)
    else:
        model.init_from_phantom_dump(dump, 1.0)

    # Print infos
    if do_print and shamrock.sys.world_rank() == 0:
        print(" - Shamrock solver config:")
        model.get_current_config().print_status()

        if in_params is not None:
            print(" - Phantom input file parameters:")
            for key, value in in_params.items():
                print(f"{key}: {value}")

        # print("Dump state:")
        # dump.print_state()

    return ctx, model
