import argparse
import os
import utils.intel_llvm
import utils.sysinfo
import utils.envscript
import utils.amd_arch
from utils.setuparg import *
from utils.oscmd import *

NAME = "Lumi-G Intel LLVM ROCM"
PATH = "machine/lumi/standard-g/intel-llvm"

def is_intel_llvm_already_installed(installfolder):
    return os.path.isfile(installfolder + "/bin/clang++")

def setup(arg : SetupArg):
    argv = arg.argv
    builddir = arg.builddir
    shamrockdir = arg.shamrockdir
    buildtype = arg.buildtype
    pylib = arg.pylib
    lib_mode = arg.lib_mode

    print("------------------------------------------")
    print("Running env setup for : "+NAME)
    print("------------------------------------------")

    if(pylib):
        print("this env does not support --pylib")
        raise ""

    parser = argparse.ArgumentParser(prog=PATH,description= NAME+' env for Shamrock')

    args = parser.parse_args(argv)
    args.gen = "ninja"

    gen, gen_opt, cmake_gen, cmake_build_type = utils.sysinfo.select_generator(args, buildtype)

    INTELLLVM_GIT_DIR = builddir+"/.env/intel-llvm-git"
    INTELLLVM_INSTALL_DIR = builddir + "/.env/intel-llvm-installdir"

    ENV_SCRIPT_PATH = builddir+"/activate"

    ENV_SCRIPT_HEADER = ""
    ENV_SCRIPT_HEADER += "export SHAMROCK_DIR="+shamrockdir+"\n"
    ENV_SCRIPT_HEADER += "export BUILD_DIR="+builddir+"\n"
    ENV_SCRIPT_HEADER += "export INTELLLVM_GIT_DIR="+INTELLLVM_GIT_DIR+"\n"
    ENV_SCRIPT_HEADER += "export INTELLLVM_INSTALL_DIR="+INTELLLVM_INSTALL_DIR+"\n"

    run_cmd("mkdir -p "+builddir+"/.env")

    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export CMAKE_GENERATOR=\""+cmake_gen+"\"\n"
    ENV_SCRIPT_HEADER += "\n"
    ENV_SCRIPT_HEADER += "export MAKE_EXEC="+gen+"\n"
    ENV_SCRIPT_HEADER += "export MAKE_OPT=("+gen_opt+")\n"
    cmake_extra_args = ""
    ENV_SCRIPT_HEADER += "export CMAKE_OPT=("+cmake_extra_args+")\n"
    ENV_SCRIPT_HEADER += "export SHAMROCK_BUILD_TYPE=\""+cmake_build_type+"\"\n"
    ENV_SCRIPT_HEADER += "\n"

    # Get current file path
    cur_file = os.path.realpath(os.path.expanduser(__file__))
    source_file = "env_built_intel-llvm.sh"
    source_path = os.path.abspath(os.path.join(cur_file, "../"+source_file))

    run_cmd("mkdir -p "+builddir)
    utils.envscript.write_env_file(
        source_path = source_path,
        header = ENV_SCRIPT_HEADER,
        path_write = ENV_SCRIPT_PATH)
