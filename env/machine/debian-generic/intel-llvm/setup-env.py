import argparse
import os

import utils.amd_arch
import utils.cuda_arch
import utils.envscript
import utils.intel_llvm
import utils.sysinfo
from utils.oscmd import *
from utils.setuparg import *

NAME = "Debian generic Intel LLVM"
PATH = "machine/debian-generic/intel-llvm"


def setup(arg: SetupArg, envgen: EnvGen):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    lib_mode = arg.lib_mode

    parser = argparse.ArgumentParser(prog=PATH, description=NAME + " env for Shamrock")

    parser.add_argument("--target", action="store", help="sycl backend to use")
    parser.add_argument("--gen", action="store", help="generator to use (ninja or make)")

    parser.add_argument("--cuda", action="store_true", help="build intel llvm with cuda support")
    parser.add_argument("--cuda-path", action="store", help="set cuda path")

    parser.add_argument("--hip", action="store_true", help="build intel llvm with hip support")
    parser.add_argument("--rocm-path", action="store", help="set ROCM path")

    args = parser.parse_args(argv)

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    configure_args = utils.intel_llvm.get_llvm_configure_arg(args)
    shamcxx_args = utils.intel_llvm.get_intel_llvm_target_flags(args)

    cmake_extra_args = ""
    if lib_mode == "shared":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=On"
    elif lib_mode == "object":
        cmake_extra_args += " -DSHAMROCK_USE_SHARED_LIB=Off"

    envgen.export_list = {
        "SHAMROCK_DIR": shamrockdir,
        "BUILD_DIR": builddir,
        "CMAKE_GENERATOR": cmake_gen,
        "MAKE_EXEC": gen,
        "MAKE_OPT": f"({gen_opt})",
        "INTEL_LLVM_CONFIGURE_ARGS": f"({configure_args})",
        "CMAKE_OPT": f"({cmake_extra_args})",
        "SHAMROCK_BUILD_TYPE": f"'{cmake_build_type}'",
        "SHAMROCK_CXX_FLAGS": f"'{shamcxx_args}'",
    }

    envgen.ext_script_list = [
        shamrockdir + "/env/helpers/clone-intel-llvm.sh",
        shamrockdir + "/env/helpers/pull_reffiles.sh",
    ]

    envgen.gen_env_file("env_built_intel-llvm.sh")

    envgen.copy_file(shamrockdir + "/env/helpers/_pysetup.py", "setup.py")
