"""Minimal hello-world tests gated on MPI world size."""

from conftest import require_world_size


def test_hello_world_size_1() -> None:
    require_world_size(1)
    print("hello world")


def test_hello_world_size_2() -> None:
    require_world_size(2)
    print("hello world")
