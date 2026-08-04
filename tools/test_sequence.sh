#!/usr/bin/env bash
set -euo pipefail
: "${SYCLCFG:?Error: SYCLCFG environment variable is not set}"
: "${MPIARGS:=}"

export LLVM_PROFILE_FILE="utests_%p.profraw"

#############################################
# Test help messages
#############################################

echo "::group::Shamrock help"
./shamrock --help
echo "::endgroup::"

echo "::group::Shamrock Test help"
./shamrock_test --help
echo "::endgroup::"

#############################################
# Test colored output
#############################################

echo "::group::Shamrock Test colored output"

echo "running: ./shamrock --help --color"
./shamrock --help --color | grep "color = enabled" || exit 1

echo "running: ./shamrock --help --nocolor"
./shamrock --help --nocolor | grep "color = disabled" || exit 1

echo "running: ./shamrock_test --help --color"
./shamrock_test --help --color | grep "color = enabled" || exit 1

echo "running: ./shamrock_test --help --nocolor"
./shamrock_test --help --nocolor | grep "color = disabled" || exit 1

echo "running: CLICOLOR_FORCE=1 ./shamrock --help"
CLICOLOR_FORCE=1 ./shamrock --help | grep "color = enabled" || exit 1

echo "running: NO_COLOR=1 ./shamrock --help"
NO_COLOR=1 ./shamrock --help | grep "color = disabled" || exit 1

echo "running: CLICOLOR_FORCE=1 NO_COLOR=1 ./shamrock --help (expect failure)"
if env CLICOLOR_FORCE=1 NO_COLOR=1 ./shamrock --help; then
    echo "Error: expected non-zero exit when CLICOLOR_FORCE and NO_COLOR are both set" >&2
    exit 1
fi

echo "running: ./shamrock --color (expect \\x1b\\[36m in output)"
./shamrock --color | grep -F $'\x1b[36m' || exit 1

echo "running: ./shamrock --nocolor (expect no \\x1b\\[36m in output)"
if ./shamrock --nocolor | grep -F $'\x1b[36m'; then
    echo "Error: expected no ANSI cyan escape sequence in output" >&2
    exit 1
fi

echo "::endgroup::"

#############################################
# Run the unittests for different world sizes
#############################################

if ! python3 -c "import pytest" 2>/dev/null; then
    python3 -m pip install pytest
fi

export PYTHONPATH="$(pwd):$(pwd)/../src/pylib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for world_size in 1 2 3 4; do
    echo "::group::Shamrock Unittests world_size = ${world_size}"

    mpirun ${MPIARGS} -n ${world_size} ./shamrock_test --smi-full --sycl-cfg "${SYCLCFG}" --unittest --loglevel 0

    echo "::group::Shamrock Python Unittests world_size = ${world_size}"
    mpirun ${MPIARGS} -n ${world_size} python3 -m pytest ../src/tests/python -v -s
    echo "::endgroup::"

    echo "::endgroup::"
done

#############################################
# Python integration
#############################################
