"""Pytest fixtures and helpers for Shamrock Python unit tests."""

import os

import pytest

import shamrock


def _init_shamrock() -> None:
    if shamrock.sys.is_initialized():
        return
    shamrock.change_loglevel(1)
    shamrock.sys.init(os.environ.get("SYCLCFG", "0:0"))


@pytest.fixture(scope="session", autouse=True)
def shamrock_session() -> None:
    """Initialize Shamrock once per pytest session on each MPI rank."""
    _init_shamrock()


def require_world_size(expected: int) -> None:
    """Skip the current test unless shamrock.sys.world_size() matches expected."""
    actual = shamrock.sys.world_size()
    if actual != expected:
        pytest.skip(f"requires world_size={expected}, got {actual}")


# Source - https://stackoverflow.com/a/38020555
# Posted by Li Feng, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-04, License - CC BY-SA 3.0


def injection_function_test() -> None:
    print("test_test_injection")


def pytest_collection_modifyitems(session, config, items):
    """called after collection has been performed, may filter or re-order
    the items in-place."""

    item = pytest.Function.from_parent(
        session,
        name="generated_test",
        callobj=injection_function_test,
    )

    items.append(item)
