"""
PEP 517 build backend building a Shamrock wheel through the env/ machinery.

A machine supporting pip builds ships a ``pyproject.toml`` next to its ``setup-env.py``
together with a ``_pip_backend.py`` shim importing this module. ``pip install <machine dir>``
then:

1. runs ``env/new-env`` for that machine in a build directory (a temporary one unless
   ``-C builddir=<path>`` is passed),
2. sources the generated ``activate`` and runs ``shamconfigure`` + ``shammake install``
   into a staging directory,
3. bundles the runtime libraries listed in ``[tool.shamrock-env] bundle-libs``
   (e.g. AdaptiveCpp's) next to the Shamrock libraries,
4. strips the absolute (build-time) entries from the RUNPATHs (requires ``patchelf``),
5. packs everything into a wheel.

The wheel uses the same layout as the legacy ``pip install .`` from a build directory
(see ``env/helpers/_pysetup.py``):

* ``site-packages/shamrock/``: python package + ``pyshamrock`` extension
* ``<prefix>/lib/``: Shamrock & bundled shared libraries (RPATH ``$ORIGIN/../../..`` from the
  extension, ``$ORIGIN/../lib`` from the executable)
* ``<prefix>/bin/shamrock``: the executable

Supported ``[tool.shamrock-env]`` keys:

* ``machine``: machine name passed to ``new-env --machine``
* ``env-args``: arguments forwarded to the machine setup (after ``--``)
* ``build-type``: ``new-env --type`` (default ``release``)
* ``mpi-dist``: name of a python distribution providing MPI (e.g. ``openmpi``), used as MPI_HOME
* ``bundle-libs``: directories (expanded after sourcing ``activate``) whose shared libraries
  are copied into the wheel's ``lib`` directory, preserving their sub-directory layout

Supported config settings (``pip install -C key=value``):

* ``builddir``: build directory to use (kept after the build, allows incremental rebuilds when
  used with ``--no-build-isolation``). By default a temporary directory is used and removed
  once the wheel is built
* ``jobs``: parallel build jobs, forwarded to the generator through ``MAKE_OPT``
"""

import base64
import csv
import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    import tomli as tomllib

SHAMROCK_DIR = Path(__file__).resolve().parents[2]


def _log(msg):
    print(f"-- [shamrock pip backend] {msg}", flush=True)


def _load_pyproject():
    with open("pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _shamrock_version():
    cmakelists = (SHAMROCK_DIR / "CMakeLists.txt").read_text()
    parts = []
    for key in ["MAJOR", "MINOR", "PATCH"]:
        m = re.search(rf"set\(SHAMROCK_VERSION_{key}\s+(\d+)\)", cmakelists)
        if m is None:
            raise RuntimeError(f"could not find SHAMROCK_VERSION_{key} in CMakeLists.txt")
        parts.append(m.group(1))
    return ".".join(parts)


def _project_info(pyproject):
    project = pyproject.get("project", {})
    return {
        "name": project.get("name", "shamrock"),
        "version": _shamrock_version(),
        "summary": project.get("description", ""),
        "requires_python": project.get("requires-python"),
        "dependencies": project.get("dependencies", []),
    }


def _wheel_tag():
    impl = sys.implementation.name
    if impl != "cpython":
        raise RuntimeError(f"unsupported python implementation: {impl}")
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"{py}-{py}-{plat}"


def _dist_name(name):
    return re.sub(r"[-_.]+", "_", name).lower()


def _metadata(info):
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {info['name']}",
        f"Version: {info['version']}",
    ]
    if info["summary"]:
        lines.append(f"Summary: {info['summary']}")
    if info["requires_python"]:
        lines.append(f"Requires-Python: {info['requires_python']}")
    for dep in info["dependencies"]:
        lines.append(f"Requires-Dist: {dep}")
    return "\n".join(lines) + "\n"


def _wheel_file(tag):
    return (
        f"Wheel-Version: 1.0\nGenerator: shamrock-pip-backend\nRoot-Is-Purelib: false\nTag: {tag}\n"
    )


def _dist_prefix(dist_name):
    """Install prefix of a python distribution shipping a bin/mpicc (e.g. the openmpi wheel)"""
    from importlib import metadata

    dist = metadata.distribution(dist_name)
    for f in dist.files or []:
        if f.name == "mpicc" and f.parent.name == "bin":
            return Path(dist.locate_file(f)).resolve().parent.parent
    raise RuntimeError(f"could not find bin/mpicc in the files of the '{dist_name}' distribution")


def _run_in_env(builddir, cmds, env, capture=False):
    script = "\n".join(["set -e", "source ./activate"] + cmds)
    if capture:
        return subprocess.run(
            ["bash", "-c", script],
            cwd=builddir,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    subprocess.run(["bash", "-c", script], cwd=builddir, env=env, check=True)


def _copy_shared_libs(srcdir, destdir):
    srcdir = Path(srcdir)
    if not srcdir.is_dir():
        raise RuntimeError(f"bundle-libs directory does not exist: {srcdir}")
    for f in srcdir.rglob("*"):
        # unversioned development symlinks (libfoo.so -> libfoo.so.1.2.3) are only used at link
        # time, the versioned file they point to is bundled
        if f.is_symlink() and f.name.endswith(".so"):
            continue
        if f.is_file() and re.search(r"\.so(\.\d+)*$", f.name):
            dest = Path(destdir) / f.relative_to(srcdir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # copy the content (wheels can not hold symlinks)
            shutil.copy2(f, dest, follow_symlinks=True)
            _log(f"bundled {f} -> {dest}")


def _is_elf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


def _sanitize_rpaths(root, env):
    """Keep only the $ORIGIN relative RUNPATH entries of the ELF files in root

    The build leaves absolute paths in the RUNPATHs (pip's temporary build environment, the
    AdaptiveCpp & Boost install directories in the build directory, ...). They must not end up in
    the wheel: everything needed at runtime is found relative to $ORIGIN.
    """
    patchelf = shutil.which("patchelf", path=env["PATH"])
    if patchelf is None:
        raise RuntimeError("patchelf is required to fix the RUNPATHs of the wheel libraries")

    for f in sorted(Path(root).rglob("*")):
        if f.is_symlink() or not f.is_file() or not _is_elf(f):
            continue
        rpath = subprocess.run(
            [patchelf, "--print-rpath", str(f)], check=True, stdout=subprocess.PIPE, text=True
        ).stdout.strip()
        entries = [e for e in rpath.split(":") if e]
        keep = list(dict.fromkeys(e for e in entries if e.startswith("$ORIGIN")))
        if keep == entries:
            continue
        if keep:
            subprocess.run([patchelf, "--set-rpath", ":".join(keep), str(f)], check=True)
        else:
            subprocess.run([patchelf, "--remove-rpath", str(f)], check=True)
        _log(f"RUNPATH of {f.name}: {':'.join(keep) or '<removed>'}")


def _build(config_settings, builddir, stagedir):
    config_settings = config_settings or {}
    pyproject = _load_pyproject()
    cfg = pyproject["tool"]["shamrock-env"]

    _log(f"build directory: {builddir}")

    env = dict(os.environ)

    # With --no-build-isolation, the scripts dir of the environment (where pip installs cmake, ninja,
    # patchelf, ...) is not necessarily in PATH. Append it, so that the tools of an isolated build
    # environment (already in PATH) still take precedence.
    scripts_dir = sysconfig.get_path("scripts")
    env["PATH"] = env.get("PATH", "") + os.pathsep + scripts_dir
    if cfg.get("mpi-dist"):
        mpi_home = _dist_prefix(cfg["mpi-dist"])
        _log(f"using MPI from the '{cfg['mpi-dist']}' distribution: {mpi_home}")
        env["MPI_HOME"] = str(mpi_home)
        env["PATH"] = str(mpi_home / "bin") + os.pathsep + env.get("PATH", "")

    if not (builddir / "activate").is_file():
        cmd = [
            sys.executable,
            str(SHAMROCK_DIR / "env" / "new-env"),
            "--machine",
            cfg["machine"],
            "--builddir",
            str(builddir),
            "--type",
            cfg.get("build-type", "release"),
            "--",
        ] + list(cfg.get("env-args", []))
        _log("running: " + shlex.join(cmd))
        subprocess.run(cmd, cwd=SHAMROCK_DIR, env=env, check=True)

    datadir = stagedir / "data"
    platlib = stagedir / "platlib"

    cmake_cmd = [
        "cmake",
        ".",
        f"-DCMAKE_INSTALL_PREFIX={datadir}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_INSTALL_BINDIR=bin",
        f"-DCMAKE_INSTALL_PYTHONDIR={platlib}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DSHAMROCK_PATCH_LIB_RPATH=On",
        "-DSHAMROCK_PYLIB_ADD_SOURCE_DIR=Off",
        "-DSHAMROCK_PYLIB_ADD_INSTALL_DIR=Off",
        "-DBUILD_TEST=Off",
    ]

    make_cmd = "shammake install"
    if config_settings.get("jobs"):
        make_cmd = f"MAKE_OPT+=(-j {int(config_settings['jobs'])}) && " + make_cmd

    _run_in_env(builddir, ["shamconfigure", shlex.join(cmake_cmd), make_cmd], env)

    # Bundle runtime libraries (e.g. AdaptiveCpp's) next to the Shamrock ones
    bundle = cfg.get("bundle-libs", [])
    if bundle:
        out = _run_in_env(
            builddir,
            ['echo "@@SHAMROCK_BUNDLE@@"'] + [f'echo "{d}"' for d in bundle],
            env,
            capture=True,
        )
        dirs = out.split("@@SHAMROCK_BUNDLE@@", 1)[1].split()
        for d in dirs:
            _copy_shared_libs(d, datadir / "lib")

    _sanitize_rpaths(datadir, env)
    _sanitize_rpaths(platlib, env)


def _record_hash(data):
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _pack_wheel(stagedir, info, wheel_directory):
    name = _dist_name(info["name"])
    version = info["version"]
    tag = _wheel_tag()
    wheel_name = f"{name}-{version}-{tag}.whl"
    distinfo = f"{name}-{version}.dist-info"
    datainfo = f"{name}-{version}.data"

    entries = []  # (archive path, file path)
    platlib = stagedir / "platlib"
    for f in sorted(platlib.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            entries.append((f.relative_to(platlib).as_posix(), f))

    datadir = stagedir / "data"
    for f in sorted(datadir.rglob("*")):
        rel = f.relative_to(datadir)
        # headers and cmake configs are not needed to run shamrock
        if rel.parts[0] == "include" or (len(rel.parts) > 1 and rel.parts[1] == "cmake"):
            continue
        if f.is_file():
            entries.append((f"{datainfo}/data/{rel.as_posix()}", f))

    records = []
    wheel_path = Path(wheel_directory) / wheel_name
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        def add(arcname, data, mode=0o644):
            zi = zipfile.ZipInfo(arcname, date_time=(2020, 1, 1, 0, 0, 0))
            zi.external_attr = (0o100000 | mode) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)
            records.append((arcname, _record_hash(data), str(len(data))))

        for arcname, f in entries:
            add(arcname, f.read_bytes(), f.stat().st_mode & 0o777)

        add(f"{distinfo}/METADATA", _metadata(info).encode())
        add(f"{distinfo}/WHEEL", _wheel_file(tag).encode())
        license_file = SHAMROCK_DIR / "LICENSE"
        if license_file.is_file():
            add(f"{distinfo}/LICENSE", license_file.read_bytes())

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for r in records:
            writer.writerow(r)
        writer.writerow((f"{distinfo}/RECORD", "", ""))
        zi = zipfile.ZipInfo(f"{distinfo}/RECORD", date_time=(2020, 1, 1, 0, 0, 0))
        zi.external_attr = (0o100000 | 0o644) << 16
        zf.writestr(zi, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED)

    _log(f"wheel written: {wheel_path}")
    return wheel_name


####################################################################################################
# PEP 517 hooks
####################################################################################################


def get_requires_for_build_wheel(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    info = _project_info(_load_pyproject())
    distinfo = Path(metadata_directory) / f"{_dist_name(info['name'])}-{info['version']}.dist-info"
    distinfo.mkdir(parents=True, exist_ok=True)
    (distinfo / "METADATA").write_text(_metadata(info))
    (distinfo / "WHEEL").write_text(_wheel_file(_wheel_tag()))
    return distinfo.name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    info = _project_info(_load_pyproject())

    # A user provided build directory is kept (incremental rebuilds), a temporary one is removed
    if config_settings and config_settings.get("builddir"):
        builddir = Path(config_settings["builddir"]).expanduser().resolve()
        tmp_builddir = None
    else:
        tmp_builddir = tempfile.mkdtemp(prefix="shamrock-pip-build-")
        builddir = Path(tmp_builddir)

    try:
        with tempfile.TemporaryDirectory(prefix="shamrock-pip-stage-") as stagedir:
            stagedir = Path(stagedir)
            _build(config_settings, builddir, stagedir)
            return _pack_wheel(stagedir, info, wheel_directory)
    finally:
        if tmp_builddir is not None:
            _log(f"removing temporary build directory: {tmp_builddir}")
            shutil.rmtree(tmp_builddir, ignore_errors=True)
