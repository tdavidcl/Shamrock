import argparse
import os

import utils.amd_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "Lumi-G Intel AdaptiveCpp Custom LLVM"
PATH = "machine/lumi/standard-g/acpp-custom-llvm"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    lib_mode = arg.lib_mode

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    parser.add_argument("--mode", action="store", help="use adaptivecpp SMCP or SSCP")

    mode = None
    args = parser.parse_args(argv)
    if args.mode == None:
        raise "no mode specified, can be SMCP or SSCP"
    elif args.mode == "SMCP":
        mode = "SMCP"
    elif args.mode == "SSCP":
        mode = "SSCP"
    else:
        raise "unknown mode, can be SMCP or SSCP"

    args.gen = "ninja"

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    ##############################
    # Generate env script header
    ##############################
    if mode == "SMCP":
        ACPP_MODE = "SMCP\n"
    elif mode == "SSCP":
        ACPP_MODE = "SSCP\n"
    else:
        raise "unknown mode, can be SMCP or SSCP"

    envgen.export_list = {
        "SHAMROCK_DIR": shamrockdir,
        "BUILD_DIR": builddir,
        "ACPP_MODE": ACPP_MODE,
        "CMAKE_GENERATOR": cmake_gen,
        "MAKE_EXEC": gen,
        "MAKE_OPT": f"({gen_opt})",
        "SHAMROCK_BUILD_TYPE": f"'{cmake_build_type}'",
    }

    envgen.ext_script_list = [
        shamrockdir + "/env/helpers/clone-acpp.sh",
        shamrockdir + "/env/helpers/clone-llvm.sh",
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    envgen.copy_env_file("exemple_batch.sh", "exemple_batch.sh")
    envgen.gen_env_file("env_built_acpp.sh")
