import argparse
import os

import utils.acpp
import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "Conda AdaptiveCpp"
PATH = "machine/conda/acpp"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    lib_mode = arg.lib_mode

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    parser.add_argument("--backend", action="store", help="sycl backend to use")
    parser.add_argument("--arch", action="store", help="arch to build")
    parser.add_argument("--gen", action="store", help="generator to use (ninja or make)")

    args = parser.parse_args(argv)

    acpp_target = utils.acpp.get_acpp_target_env(args)
    if acpp_target == None:
        print("-- target not specified using acpp default")
    else:
        print("-- setting acpp target to :", acpp_target)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    cmake_extra_args = ""
    if lib_mode == "shared":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=On"
    elif lib_mode == "object":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=Off"

    envgen.export_list = {
        "SHAMROCK_DIR": shamrockdir,
        "BUILD_DIR": builddir,
        "MAKE_EXEC": "ninja",
        "MAKE_OPT": f"({gen_opt})",
        "CMAKE_OPT": f"({cmake_extra_args})",
        "SHAMROCK_BUILD_TYPE": f"'{cmake_build_type}'",
        "SHAMROCK_CXX_FLAGS": "\" --acpp-targets='" + acpp_target + "'\"",
    }

    envgen.ext_script_list = [
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    envgen.gen_env_file("conda_acpp_env.sh")
    envgen.copy_env_file("environment.yml", "environment.yml")
    envgen.copy_file(shamrockdir + "/env/helpers/_pysetup.py", "setup.py")
