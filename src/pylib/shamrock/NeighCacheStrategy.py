"""
Neighbour cache strategies.

This module exposes the members of the :py:class:`shamrock.NeighCacheStrategy` enum
directly, so that scripts can use the short form::

    from shamrock.NeighCacheStrategy import SingleStage, TwoStage

    cfg.set_neigh_cache_strategy(SingleStage)

rather than spelling out ``shamrock.NeighCacheStrategy.SingleStage`` at every call site.
The enum type itself is re-exported too, so ``shamrock.NeighCacheStrategy.SingleStage``
yields the same value whether the name resolves to this module or to the enum class.
"""

try:
    # try to import from the global namespace (works if embedded python interpreter is used)
    from pyshamrock import NeighCacheStrategy
except ImportError:
    # then it is a library mode, we import from the local namespace
    from .pyshamrock import NeighCacheStrategy

SingleStage = NeighCacheStrategy.SingleStage
"""Single tree traversal per particle."""

TwoStage = NeighCacheStrategy.TwoStage
"""Two stage neighbours search (see the shamrock paper), the default."""

__all__ = ["NeighCacheStrategy", "SingleStage", "TwoStage"]
