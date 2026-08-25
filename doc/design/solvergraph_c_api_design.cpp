// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file solvergraph_c_api_design.cpp
 * @brief Standalone sketch: shamsolvergraph as a C-style API -- POD structs,
 *        one ops table per type, type erasure through pointers.
 *
 * Build: g++ -std=c++20 -O2 solvergraph_c_api_design.cpp && ./a.out
 *        (self-contained, no dependencies; runs on godbolt.org)
 *
 * ---------------------------------------------------------------------------
 * THE WHOLE IDEA IN FOUR LINES
 * ---------------------------------------------------------------------------
 *
 *   struct SgEdge          { const SgEdgeOps *ops; u64 uuid; ... };  // header
 *   struct FieldSpanEdge   { SgEdge edge;      SpanList<T> spans; }; // edge  first
 *   struct FieldRefsEdge   { FieldSpanEdge<T> span; RefList<T> refs; }; // span first
 *
 * Every struct starts with the thing it "is a". So:
 *
 *   FieldRefsEdge<f64> vel;
 *   FieldSpanEdge<f64> *as_span = &vel.span;        // <-- THE DECAY. free.
 *   SgEdge             *as_edge = &vel.span.edge;   // <-- two levels. free.
 *
 * That is the whole answer to "how does a field ref decay into a field span":
 * it is the address of a subobject, spelled out, no cast, no vtable walk, no
 * dynamic_cast, checkable by the compiler. Going the other way (SgEdge* back
 * to the concrete type) is a plain pointer cast guarded by a capability bit --
 * the C replacement for dynamic_pointer_cast, see sg_edge_as_span().
 *
 * The only dynamic dispatch left is the handful of operations that genuinely
 * differ per concrete type (free_alloc, ensure_sizes, evaluate, ...). They live
 * in one ops table per type, hung off the header. Everything else -- uuid,
 * name, tex symbol, the span array itself -- is plain data at a known offset,
 * so reading it costs nothing.
 *
 *   today                                    here
 *   ------------------------------------     ------------------------------------
 *   class IEdge / IEdgeNamed / IFreeable     struct SgEdge (data) + SgEdgeOps (fn ptrs)
 *   class IFieldSpan<T> (pure interface)     struct FieldSpanEdge<T> (concrete, holds spans)
 *   class IFieldRefs<T> : IFieldSpan<T>      struct FieldRefsEdge<T> { FieldSpanEdge<T> span; ... }
 *   shared_ptr<IFieldSpan<T>>                FieldSpanEdge<T>*
 *   dynamic_pointer_cast<T>                  sg_edge_as_span<T>() -- one bit test
 *   virtual get_spans()                      e->spans        (a member)
 *   class INode                              struct SgNode + SgNodeOps
 *   vector<shared_ptr<IEdge>> ro/rw          SgEdge *ro[]; u32 n_ro;
 *   class OperationSequence : INode          struct OpSeq { SgNode node; SgNode **nodes; }
 *
 * Ownership is left to the caller on purpose (everything below is a borrowed
 * pointer, like a C API). Wrap the roots in whatever you like -- unique_ptr,
 * an arena, or plain stack objects as in main().
 */

#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f64 = double;

// ===========================================================================
// PART 0 -- plain data (stand-ins for DistributedData / PatchDataField)
// ===========================================================================

/// one patch worth of a field, borrowed
template<class T>
struct PatchSpan {
    u64 id;
    T *ptr;
    u32 size;
};

/// == DDPatchDataFieldSpanPointer<T>, as a flat array
template<class T>
struct SpanList {
    PatchSpan<T> *items;
    u32 count;
};

/// == shamrock::PatchDataField<T>
template<class T>
struct PatchField {
    T *data;
    u32 size;
    u32 capacity;
};

template<class T>
struct PatchFieldRef {
    u64 id;
    PatchField<T> *field;
};

/// == DDPatchDataFieldRef<T>
template<class T>
struct RefList {
    PatchFieldRef<T> *items;
    u32 count;
};

struct PatchSize {
    u64 id;
    u32 size;
};

/// == DistributedData<u32>
struct SizeList {
    const PatchSize *items;
    u32 count;
};

inline u64 sg_next_uuid() {
    static u64 c = 0;
    return c++;
}

/// unique address per T, used as a runtime type tag (C: an enum or a string)
template<class T>
inline const char sg_type_key = 0;

// ===========================================================================
// PART 1 -- the edge header and its ops table  (this is the whole framework)
// ===========================================================================

struct SgEdge;

/// capability bits: which prefix structs this edge actually starts with.
/// Replaces the "is it really an IFieldRefs?" question that dynamic_cast answers.
enum SgEdgeCap : u32 {
    SG_CAP_DATA = 1u << 0,
    SG_CAP_SPAN = 1u << 1,
    SG_CAP_REFS = 1u << 2, // implies SG_CAP_SPAN
};

/// One static instance per concrete edge type. NULL slot == "default / not supported",
/// the C idiom for a virtual with a body (cf. struct file_operations).
struct SgEdgeOps {
    const char *type_name;
    u32 caps;
    const void *elem_type; ///< &sg_type_key<T>, nullptr for non-field edges

    void (*free_alloc)(SgEdge *self);              ///< NULL => nothing to free
    void (*ensure_sizes)(SgEdge *self, SizeList);  ///< NULL => not resizable
    bool (*check_sizes)(const SgEdge *, SizeList); ///< NULL => always ok
};

/// The header every edge starts with (== IEdge + IEdgeNamed + WithUUID, as data)
struct SgEdge {
    const SgEdgeOps *ops;
    u64 uuid;
    const char *name;
    const char *tex;
};

// ---- the generic C API over the header ------------------------------------
// none of these dispatch: the data is right there in the header.

inline const char *sg_edge_label(const SgEdge *e) { return e->name; }
inline const char *sg_edge_tex(const SgEdge *e) { return e->tex; }
inline u64 sg_edge_uuid(const SgEdge *e) { return e->uuid; }
inline const char *sg_edge_type_name(const SgEdge *e) { return e->ops->type_name; }

inline void sg_edge_init(SgEdge *e, const SgEdgeOps *ops, const char *name, const char *tex) {
    e->ops  = ops;
    e->uuid = sg_next_uuid();
    e->name = name;
    e->tex  = tex;
}

/// == IFreeable::free_alloc, with a default no-op
inline void sg_edge_free_alloc(SgEdge *e) {
    if (e->ops->free_alloc != nullptr) {
        e->ops->free_alloc(e);
    }
}

/// == IFieldSpan::ensure_sizes. The one call that genuinely needs dispatch:
/// a span edge cannot grow, a refs edge can (it owns the fields).
inline void sg_edge_ensure_sizes(SgEdge *e, SizeList sizes) {
    if (e->ops->ensure_sizes != nullptr) {
        e->ops->ensure_sizes(e, sizes);
    } else if (e->ops->check_sizes != nullptr && !e->ops->check_sizes(e, sizes)) {
        std::printf("FATAL: size mismatch on edge '%s'\n", e->name);
        std::abort();
    }
}

// ===========================================================================
// PART 2 -- concrete edges. Plain structs, each starting with what it "is a".
// ===========================================================================

// ------------------------------------------------------------- DataEdge ---
/// == solvergraph::IDataEdge<T>
template<class T>
struct DataEdge {
    SgEdge edge; ///< MUST be first
    T data;
};

template<class T>
void data_edge_free_alloc(SgEdge *self) {
    reinterpret_cast<DataEdge<T> *>(self)->data = T{};
}

template<class T>
inline const SgEdgeOps data_edge_ops
    = {"DataEdge", SG_CAP_DATA, &sg_type_key<T>, &data_edge_free_alloc<T>, nullptr, nullptr};

template<class T>
void data_edge_init(DataEdge<T> *e, const char *name, const char *tex, T value) {
    sg_edge_init(&e->edge, &data_edge_ops<T>, name, tex);
    e->data = value;
}

// -------------------------------------------------------- FieldSpanEdge ---
/// == solvergraph::IFieldSpan<T> / FieldSpan<T>.
/// Note it is NOT an interface any more: the span list is plain data, so
/// "get_spans()" is `e->spans`, no virtual call.
template<class T>
struct FieldSpanEdge {
    SgEdge edge; ///< MUST be first
    SpanList<T> spans;
};

template<class T>
bool field_span_check_sizes(const SgEdge *self, SizeList sizes) {
    auto *e = reinterpret_cast<const FieldSpanEdge<T> *>(self);
    if (e->spans.count != sizes.count) {
        return false;
    }
    for (u32 i = 0; i < sizes.count; i++) {
        if (e->spans.items[i].id != sizes.items[i].id
            || e->spans.items[i].size != sizes.items[i].size) {
            return false;
        }
    }
    return true;
}

template<class T>
void field_span_free_alloc(SgEdge *self) {
    reinterpret_cast<FieldSpanEdge<T> *>(self)->spans = SpanList<T>{nullptr, 0};
}

template<class T>
inline const SgEdgeOps field_span_edge_ops
    = {"FieldSpanEdge",
       SG_CAP_SPAN,
       &sg_type_key<T>,
       &field_span_free_alloc<T>,
       nullptr, ///< a span edge does not own the memory: it cannot resize
       &field_span_check_sizes<T>};

template<class T>
void field_span_edge_init(
    FieldSpanEdge<T> *e, const char *name, const char *tex, SpanList<T> spans) {
    sg_edge_init(&e->edge, &field_span_edge_ops<T>, name, tex);
    e->spans = spans;
}

// -------------------------------------------------------- FieldRefsEdge ---
/// == solvergraph::IFieldRefs<T> / FieldRefs<T>.
/// It starts with a FieldSpanEdge<T>, so it *is* one: that is the decay.
template<class T>
struct FieldRefsEdge {
    FieldSpanEdge<T> span; ///< MUST be first -- this is the "inherits from" part
    RefList<T> refs;
    PatchSpan<T> *span_storage; ///< backing array kept in sync with refs
};

template<class T>
void field_refs_sync_spans(FieldRefsEdge<T> *e) {
    for (u32 i = 0; i < e->refs.count; i++) {
        PatchField<T> *f   = e->refs.items[i].field;
        e->span_storage[i] = PatchSpan<T>{e->refs.items[i].id, f->data, f->size};
    }
    e->span.spans = SpanList<T>{e->span_storage, e->refs.count};
}

template<class T>
void field_refs_free_alloc(SgEdge *self) {
    auto *e       = reinterpret_cast<FieldRefsEdge<T> *>(self);
    e->refs       = RefList<T>{nullptr, 0};
    e->span.spans = SpanList<T>{nullptr, 0};
}

/// unlike a span edge, this one owns the fields, so it can actually resize.
/// A node holding only a FieldSpanEdge<T>* still reaches this through the ops
/// table -- that is the dispatch we deliberately kept.
template<class T>
void field_refs_ensure_sizes(SgEdge *self, SizeList sizes) {
    auto *e    = reinterpret_cast<FieldRefsEdge<T> *>(self);
    bool dirty = false;
    for (u32 i = 0; i < sizes.count; i++) {
        PatchField<T> *f = nullptr;
        for (u32 j = 0; j < e->refs.count; j++) {
            if (e->refs.items[j].id == sizes.items[i].id) {
                f = e->refs.items[j].field;
            }
        }
        if (f == nullptr) {
            std::printf(
                "FATAL: missing field ref on '%s' at patch %llu\n",
                e->span.edge.name,
                (unsigned long long) sizes.items[i].id);
            std::abort();
        }
        if (f->size != sizes.items[i].size) {
            if (sizes.items[i].size > f->capacity) {
                std::printf("FATAL: cannot grow '%s' past capacity\n", e->span.edge.name);
                std::abort();
            }
            f->size = sizes.items[i].size;
            dirty   = true;
        }
    }
    if (dirty) {
        field_refs_sync_spans(e);
    }
}

template<class T>
inline const SgEdgeOps field_refs_edge_ops
    = {"FieldRefsEdge",
       SG_CAP_SPAN | SG_CAP_REFS,
       &sg_type_key<T>,
       &field_refs_free_alloc<T>,
       &field_refs_ensure_sizes<T>,
       &field_span_check_sizes<T>}; ///< reused as-is: the span prefix is at offset 0

/// The C constructor chain: initialise the prefix, then override the ops table.
template<class T>
void field_refs_edge_init(
    FieldRefsEdge<T> *e,
    const char *name,
    const char *tex,
    RefList<T> refs,
    PatchSpan<T> *span_storage) {
    field_span_edge_init(&e->span, name, tex, SpanList<T>{nullptr, 0});
    e->span.edge.ops = &field_refs_edge_ops<T>; ///< <-- "override"
    e->refs          = refs;
    e->span_storage  = span_storage;
    field_refs_sync_spans(e);
}

// ===========================================================================
// PART 3 -- THE DECAY, both directions
// ===========================================================================
//
// Downward (concrete -> more generic) is free and needs no API at all:
//
//     FieldRefsEdge<f64> vel;
//     FieldSpanEdge<f64> *s = &vel.span;        // IFieldRefs<T>& -> IFieldSpan<T>&
//     SgEdge             *e = &vel.span.edge;   // ...            -> IEdge&
//
// Upward (SgEdge* -> concrete) is a pointer cast guarded by the capability
// bits + the element type tag. This is the dynamic_pointer_cast replacement:
// one load, one AND, one compare -- and it returns NULL instead of throwing.
//
// The casts are legal because every struct is standard-layout with its parent
// as the first member, so the parent subobject and the object share an address.

static_assert(std::is_standard_layout_v<SgEdge>);
static_assert(std::is_standard_layout_v<FieldSpanEdge<f64>>);
static_assert(std::is_standard_layout_v<FieldRefsEdge<f64>>);
static_assert(offsetof(FieldSpanEdge<f64>, edge) == 0, "SgEdge must be the first member");
static_assert(offsetof(FieldRefsEdge<f64>, span) == 0, "FieldSpanEdge must be the first member");

template<class T>
inline FieldSpanEdge<T> *sg_edge_as_span(SgEdge *e) {
    if (e == nullptr || (e->ops->caps & SG_CAP_SPAN) == 0 || e->ops->elem_type != &sg_type_key<T>) {
        return nullptr;
    }
    return reinterpret_cast<FieldSpanEdge<T> *>(e);
}

template<class T>
inline FieldRefsEdge<T> *sg_edge_as_refs(SgEdge *e) {
    if (e == nullptr || (e->ops->caps & SG_CAP_REFS) == 0 || e->ops->elem_type != &sg_type_key<T>) {
        return nullptr;
    }
    return reinterpret_cast<FieldRefsEdge<T> *>(e);
}

template<class T>
inline DataEdge<T> *sg_edge_as_data(SgEdge *e) {
    if (e == nullptr || (e->ops->caps & SG_CAP_DATA) == 0 || e->ops->elem_type != &sg_type_key<T>) {
        return nullptr;
    }
    return reinterpret_cast<DataEdge<T> *>(e);
}

// ===========================================================================
// PART 4 -- the node header and its ops table  (== INode)
// ===========================================================================

struct SgNode;

struct SgNodeOps {
    const char *type_name;
    void (*evaluate)(SgNode *self);
    const char *(*tex)(const SgNode *self);                 ///< NULL => label
    void (*dot_partial)(const SgNode *self, std::string *); ///< NULL => default below
    const SgNode *(*dot_start)(const SgNode *self);         ///< NULL => self
    const SgNode *(*dot_end)(const SgNode *self);           ///< NULL => self
};

struct SgNode {
    const SgNodeOps *ops;
    u64 uuid;
    const char *label;
    SgEdge **ro; ///< borrowed array, owned by the concrete node
    u32 n_ro;
    SgEdge **rw;
    u32 n_rw;
};

inline void sg_node_init(SgNode *n, const SgNodeOps *ops, const char *label) {
    n->ops   = ops;
    n->uuid  = sg_next_uuid();
    n->label = label;
    n->ro    = nullptr;
    n->n_ro  = 0;
    n->rw    = nullptr;
    n->n_rw  = 0;
}

inline void sg_node_evaluate(SgNode *n) { n->ops->evaluate(n); }

inline const char *sg_node_tex(const SgNode *n) {
    return n->ops->tex != nullptr ? n->ops->tex(n) : n->label;
}

inline const SgNode *sg_node_dot_start(const SgNode *n) {
    return n->ops->dot_start != nullptr ? n->ops->dot_start(n) : n;
}
inline const SgNode *sg_node_dot_end(const SgNode *n) {
    return n->ops->dot_end != nullptr ? n->ops->dot_end(n) : n;
}

/// default _impl_get_dot_graph_partial: pure header walking, no dispatch
inline void sg_node_dot_partial(const SgNode *n, std::string *out) {
    if (n->ops->dot_partial != nullptr) {
        n->ops->dot_partial(n, out);
        return;
    }
    char buf[512];
    std::snprintf(
        buf, sizeof(buf), "n_%llu [label=\"%s\"];\n", (unsigned long long) n->uuid, n->label);
    *out += buf;
    for (u32 i = 0; i < n->n_ro; i++) {
        std::snprintf(
            buf,
            sizeof(buf),
            "e_%llu -> n_%llu [style=\"dashed\", color=green];\n"
            "e_%llu [label=\"%s\",shape=rect];\n",
            (unsigned long long) n->ro[i]->uuid,
            (unsigned long long) n->uuid,
            (unsigned long long) n->ro[i]->uuid,
            n->ro[i]->name);
        *out += buf;
    }
    for (u32 i = 0; i < n->n_rw; i++) {
        std::snprintf(
            buf,
            sizeof(buf),
            "n_%llu -> e_%llu [style=\"dashed\", color=red];\n"
            "e_%llu [label=\"%s\",shape=rect];\n",
            (unsigned long long) n->uuid,
            (unsigned long long) n->rw[i]->uuid,
            (unsigned long long) n->rw[i]->uuid,
            n->rw[i]->name);
        *out += buf;
    }
}

/// == INode::print_node_info
inline void sg_node_print_info(const SgNode *n) {
    std::printf("Node info :\n");
    std::printf(" - Node type : %s\n", n->ops->type_name);
    std::printf(" - Node UUID : %llu\n", (unsigned long long) n->uuid);
    std::printf(" - Node label : %s\n", n->label);
    auto dump = [](const char *title, SgEdge **e, u32 c) {
        std::printf(" - %s: %u\n", title, c);
        for (u32 i = 0; i < c; i++) {
            std::printf(
                "     - uuid = %llu, label = %s, type = %s\n",
                (unsigned long long) e[i]->uuid,
                e[i]->name,
                e[i]->ops->type_name);
        }
    };
    dump("Node Read Only edges", n->ro, n->n_ro);
    dump("Node Read Write edges", n->rw, n->n_rw);
}

// ===========================================================================
// PART 5 -- a real node: ForwardEuler
// ===========================================================================
//
// Compare with shammodels::common::modules::ForwardEuler:
//   X_RO(IDataEdge<Tscal>, dt)           -> const DataEdge<T>      *dt
//   X_RO(IFieldSpan<T>, time_derivative) -> const FieldSpanEdge<T> *time_derivative
//   X_RW(IFieldSpan<T>, field)           ->       FieldSpanEdge<T> *field
//
// The slots are typed pointers, so get_edges() disappears entirely and no cast
// happens at evaluate() time. `const` on the RO slots gives the same read-only
// guarantee the `const IFieldSpan<T>&` in the Edges struct gives today.

template<class T>
struct ForwardEuler {
    SgNode node; ///< MUST be first
    u32 nvar;

    const DataEdge<T> *dt;
    const FieldSpanEdge<T> *time_derivative;
    FieldSpanEdge<T> *field;

    SgEdge *ro_slots[2];
    SgEdge *rw_slots[1];
};

template<class T>
void forward_euler_evaluate(SgNode *self) {
    auto *n = reinterpret_cast<ForwardEuler<T> *>(self);

    // ensure_sizes on the rw field, using the ro field sizes
    PatchSize sizes[16];
    const SpanList<T> &src = n->time_derivative->spans;
    for (u32 i = 0; i < src.count; i++) {
        sizes[i] = PatchSize{src.items[i].id, src.items[i].size};
    }
    sg_edge_ensure_sizes(&n->field->edge, SizeList{sizes, src.count});

    const T dt = n->dt->data;
    for (u32 i = 0; i < n->field->spans.count; i++) {
        PatchSpan<T> &dst = n->field->spans.items[i];
        PatchSpan<T> &s   = src.items[i];
        for (u32 k = 0; k < dst.size * n->nvar; k++) {
            dst.ptr[k] = dst.ptr[k] + dt * s.ptr[k];
        }
    }
}

template<class T>
const char *forward_euler_tex(const SgNode *) {
    return "$f \\mathrel{+}= \\Delta t\\,\\dot f$";
}

template<class T>
inline const SgNodeOps forward_euler_ops
    = {"ForwardEuler",
       &forward_euler_evaluate<T>,
       &forward_euler_tex<T>,
       nullptr, ///< default dot_partial
       nullptr, ///< default dot_start
       nullptr};

template<class T>
void forward_euler_init(ForwardEuler<T> *n, u32 nvar) {
    sg_node_init(&n->node, &forward_euler_ops<T>, "ForwardEuler");
    n->nvar = nvar;
}

/// == EXPAND_NODE_EDGES' set_edges. Passing the wrong edge type is now a
/// compile error here instead of a throw inside evaluate().
template<class T>
void forward_euler_set_edges(
    ForwardEuler<T> *n,
    const DataEdge<T> *dt,
    const FieldSpanEdge<T> *time_derivative,
    FieldSpanEdge<T> *field) {
    n->dt              = dt;
    n->time_derivative = time_derivative;
    n->field           = field;

    // the erased view used by the graph walkers (dot, info, free_alloc)
    n->ro_slots[0] = const_cast<SgEdge *>(&dt->edge);
    n->ro_slots[1] = const_cast<SgEdge *>(&time_derivative->edge);
    n->rw_slots[0] = &field->edge;
    n->node.ro     = n->ro_slots;
    n->node.n_ro   = 2;
    n->node.rw     = n->rw_slots;
    n->node.n_rw   = 1;
}

// ===========================================================================
// PART 6 -- OperationSequence
// ===========================================================================

struct OpSeq {
    SgNode node; ///< MUST be first
    SgNode **nodes;
    u32 count;
};

inline void op_seq_evaluate(SgNode *self) {
    auto *s = reinterpret_cast<OpSeq *>(self);
    for (u32 i = 0; i < s->count; i++) {
        sg_node_evaluate(s->nodes[i]);
    }
}

inline const char *op_seq_tex(const SgNode *self) {
    // (a real one would build the composite string; kept short here)
    return self->label;
}

inline void op_seq_dot_partial(const SgNode *self, std::string *out) {
    auto *s = reinterpret_cast<const OpSeq *>(self);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "subgraph cluster_%llu {\n", (unsigned long long) self->uuid);
    *out += buf;
    for (u32 i = 0; i < s->count; i++) {
        sg_node_dot_partial(s->nodes[i], out);
    }
    for (u32 i = 0; i + 1 < s->count; i++) {
        std::snprintf(
            buf,
            sizeof(buf),
            "n_%llu -> n_%llu [weight=3];\n",
            (unsigned long long) sg_node_dot_end(s->nodes[i])->uuid,
            (unsigned long long) sg_node_dot_start(s->nodes[i + 1])->uuid);
        *out += buf;
    }
    std::snprintf(buf, sizeof(buf), "label = \"%s\";\n}\n", self->label);
    *out += buf;
}

inline const SgNode *op_seq_dot_start(const SgNode *self) {
    auto *s = reinterpret_cast<const OpSeq *>(self);
    return sg_node_dot_start(s->nodes[0]);
}
inline const SgNode *op_seq_dot_end(const SgNode *self) {
    auto *s = reinterpret_cast<const OpSeq *>(self);
    return sg_node_dot_end(s->nodes[s->count - 1]);
}

inline const SgNodeOps op_seq_ops
    = {"OperationSequence",
       &op_seq_evaluate,
       &op_seq_tex,
       &op_seq_dot_partial, ///< overrides the default
       &op_seq_dot_start,
       &op_seq_dot_end};

inline void op_seq_init(OpSeq *s, const char *label, SgNode **nodes, u32 count) {
    if (count == 0) {
        std::printf("FATAL: OperationSequence must have at least one node\n");
        std::abort();
    }
    sg_node_init(&s->node, &op_seq_ops, label);
    s->nodes = nodes;
    s->count = count;
}

// ===========================================================================
// PART 7 -- demo
// ===========================================================================

static void decay_demo() {
    std::printf("== decay demo ==\n");

    f64 buf0[8]              = {1., 2., 3.};
    PatchField<f64> f0       = {buf0, 3, 8};
    PatchFieldRef<f64> rl[1] = {{0, &f0}};
    PatchSpan<f64> store[1];

    FieldRefsEdge<f64> vel;
    field_refs_edge_init(&vel, "rho", "\\rho", RefList<f64>{rl, 1}, store);

    // (a) the decay: address of a subobject. No cast, no dispatch, no runtime cost.
    FieldSpanEdge<f64> *as_span = &vel.span;
    SgEdge *as_edge             = &vel.span.edge;
    std::printf(
        "  a) &refs.span      -> FieldSpanEdge* : n=%u, spans[0].size=%u\n",
        as_span->spans.count,
        as_span->spans.items[0].size);
    std::printf(
        "  b) &refs.span.edge -> SgEdge*        : label=%s tex=%s type=%s\n",
        sg_edge_label(as_edge),
        sg_edge_tex(as_edge),
        sg_edge_type_name(as_edge));

    // (c) upward, from an erased SgEdge*: the dynamic_pointer_cast replacement
    std::printf(
        "  c) sg_edge_as_span<f64>(e)=%p  sg_edge_as_refs<f64>(e)=%p  as_data<f64>(e)=%p\n",
        (void *) sg_edge_as_span<f64>(as_edge),
        (void *) sg_edge_as_refs<f64>(as_edge),
        (void *) sg_edge_as_data<f64>(as_edge)); // NULL: caps bit not set

    // (d) a plain span edge is NOT a refs edge -> NULL, no throw
    PatchSpan<f64> sp[1] = {{0, buf0, 3}};
    FieldSpanEdge<f64> plain;
    field_span_edge_init(&plain, "v", "v", SpanList<f64>{sp, 1});
    std::printf(
        "  d) on a FieldSpanEdge: as_span=%p as_refs=%p (NULL as expected)\n",
        (void *) sg_edge_as_span<f64>(&plain.edge),
        (void *) sg_edge_as_refs<f64>(&plain.edge));

    // (e) wrong element type -> NULL too (the tag check)
    std::printf(
        "  e) sg_edge_as_span<float>(e)=%p (NULL: wrong T)\n\n",
        (void *) sg_edge_as_span<float>(as_edge));
}

int main() {
    decay_demo();

    std::printf("== graph demo ==\n");

    // --- the data ---------------------------------------------------------
    f64 vel0_buf[8] = {0., 0., 0., 0.};
    f64 vel1_buf[8] = {1., 1.};
    f64 acc0_buf[8] = {1., 2., 3., 4.};
    f64 acc1_buf[8] = {-1., -2.};

    PatchField<f64> vel0 = {vel0_buf, 4, 8};
    PatchField<f64> vel1 = {vel1_buf, 2, 8};

    // --- the edges --------------------------------------------------------
    DataEdge<f64> dt;
    data_edge_init(&dt, "dt", "\\Delta t", 0.5);

    PatchSpan<f64> acc_spans[2] = {{0, acc0_buf, 4}, {1, acc1_buf, 2}};
    FieldSpanEdge<f64> acc;
    field_span_edge_init(&acc, "acc", "a", SpanList<f64>{acc_spans, 2});

    PatchFieldRef<f64> vel_refs[2] = {{0, &vel0}, {1, &vel1}};
    PatchSpan<f64> vel_spans[2];
    FieldRefsEdge<f64> vel;
    field_refs_edge_init(&vel, "vel", "v", RefList<f64>{vel_refs, 2}, vel_spans);

    // --- the nodes --------------------------------------------------------
    ForwardEuler<f64> euler1;
    ForwardEuler<f64> euler2;
    forward_euler_init(&euler1, 1);
    forward_euler_init(&euler2, 1);

    //                                     acc is a FieldSpanEdge  ---.
    //           vel is a FieldRefsEdge, decayed by `&vel.span`  ---.  |
    forward_euler_set_edges(&euler1, &dt, &acc, &vel.span); //    <-'--'
    forward_euler_set_edges(&euler2, &dt, &acc, &vel.span);

    SgNode *seq_nodes[2] = {&euler1.node, &euler2.node};
    OpSeq seq;
    op_seq_init(&seq, "kick x2", seq_nodes, 2);

    // --- run --------------------------------------------------------------
    sg_node_evaluate(&seq.node);

    std::printf("vel patch0 =");
    for (u32 i = 0; i < vel0.size; i++) {
        std::printf(" %g", vel0.data[i]);
    }
    std::printf("   (expected 1 2 3 4)\n");
    std::printf("vel patch1 =");
    for (u32 i = 0; i < vel1.size; i++) {
        std::printf(" %g", vel1.data[i]);
    }
    std::printf("   (expected 0 -1)\n\n");

    sg_node_print_info(&euler1.node);

    std::string dot;
    sg_node_dot_partial(&seq.node, &dot);
    std::printf("\ndigraph G {\n%s}\n", dot.c_str());

    // free_alloc through the erased handle (== IFreeable)
    sg_edge_free_alloc(&vel.span.edge);
    std::printf("\nafter free_alloc: vel spans = %u\n", vel.span.spans.count);

    std::printf(
        "\nsizeof(SgEdge)=%zu  sizeof(FieldSpanEdge<f64>)=%zu  sizeof(FieldRefsEdge<f64>)=%zu\n",
        sizeof(SgEdge),
        sizeof(FieldSpanEdge<f64>),
        sizeof(FieldRefsEdge<f64>));
    return 0;
}
