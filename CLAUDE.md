@AGENTS.md

## Claude Code on the web: container setup

This container has no GPU, so AdaptiveCpp is built from source targeting
the OpenMP backend. Automated by `.claude/hooks/session-start.sh`; the
condensed steps it runs:

```bash
# System packages (Boost.context/fiber + LLVM for AdaptiveCpp, OpenMPI, pre-commit)
apt-get install -y libboost-context-dev libboost-fiber-dev llvm-18-dev \
  libclang-18-dev libomp-18-dev libopenmpi-dev openmpi-bin pre-commit

# Submodules
git submodule update --init --recursive

# Env (does NOT build AdaptiveCpp yet)
./env/new-env --machine debian-generic.acpp --builddir build -- --backend omp
```

pre-commit hook venvs also need `SETUPTOOLS_USE_DISTUTILS=stdlib` exported —
Debian's patched sysconfig scheme otherwise breaks setuptools' vendored
distutils with `AttributeError: install_layout`.

The hook deliberately stops there: `./shamenv_do shamconfigure` builds
AdaptiveCpp from source on its first invocation (a few minutes), so that
cost is paid inline the first time a build/test is actually needed rather
than blocking every session start.
