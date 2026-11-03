# Solver graph edges

Edges carry data between nodes in a solver graph. They live in
`shamrock::solvergraph` and inherit
[`IEdge`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/IEdge.hpp)
(or [`IEdgeNamed`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/IEdgeNamed.hpp)).

Every edge must implement `free_alloc()` and the label/TeX hooks.

## Derive from `IEdge`

Implement the three required overrides:

```cpp
class MyEdge : public shamrock::solvergraph::IEdge {
  public:
    // 1. DOT label
    std::string _impl_get_dot_label() const override { return "MyEdge"; }
    // 2. TeX symbol
    std::string _impl_get_tex_symbol() const override { return "e"; }
    // 3. free allocated resources
    void free_alloc() override { /* ... */ }
};
```

`get_label()` and `get_tex_symbol()` call the `_impl_*` methods.

## Derive from `IEdgeNamed`

`IEdgeNamed` stores a label and a TeX symbol, and already implements
`_impl_get_dot_label` / `_impl_get_tex_symbol`. Subclasses inherit the
constructor and only add payload + `free_alloc()`.

```cpp
class MyNamedEdge : public shamrock::solvergraph::IEdgeNamed {
  public:
    using IEdgeNamed::IEdgeNamed; // takes (label, tex_symbol)

    void free_alloc() override { /* ... */ }
};

// usage — first arg is the DOT label, second is the TeX symbol
std::string label      = "my_edge";
std::string tex_symbol = "e";
auto e = std::make_shared<MyNamedEdge>(label, tex_symbol);
```

[`IDataEdge<T>`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/IDataEdge.hpp)
is the common named edge with a `T data` payload:

```cpp
auto e = shamrock::solvergraph::IDataEdge<f64>::make_shared(label, tex_symbol);
e->data = 1.0;
```

## Reusing allocations between evaluations

An edge kept alive between steps should be resized by the node writing it
rather than recreated, so that its buffers are reused instead of being
reallocated at every step (`DeviceBuffer::resize` only reallocates when
growing beyond the capacity, or shrinking below half of it):

- per-patch fields (`Field<T>`, `FieldRefs<T>`, `PatchDataLayerEdge`) use
  `ensure_sizes(sizes)`;
- per patch-pair containers (`DistributedDataShared<T>`, held by
  `DDSharedBuffers<T>`, `PatchDataFieldDDShared<T>`,
  `PatchDataLayerDDShared`) use `reshape_like(reference, create, update)`,
  which reuses the object stored under the same key, then recycles the
  leftover objects, and only calls `create()` when none is left.

Writers then overwrite the content in place (e.g. `assign_subset_to`,
`deserialize_buf_inplace`) instead of clearing and appending. Do not call
`free_alloc()` on such edges at the end of a step, it would drop the
allocations that are meant to be reused.

## Related files

- [`IEdge.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/IEdge.hpp)
- [`IEdgeNamed.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/IEdgeNamed.hpp)
- [`IDataEdge.hpp`](../../../src/shamsolvergraph/include/shamsolvergraph/edge/IDataEdge.hpp)
- [`IEdge_tests.cpp`](../../../src/tests/shamsolvergraph/edge/IEdge_tests.cpp)
