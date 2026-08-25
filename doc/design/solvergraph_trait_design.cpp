// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file solvergraph_trait_design.cpp
 * @brief Standalone sketch: shamsolvergraph without inheritance, using
 *        Rust-style traits (explicit vtables) over plain C structs.
 *
 * Build: g++ -std=c++20 -O2 solvergraph_trait_design.cpp && ./a.out
 *        (self-contained, no dependencies; runs on godbolt.org)
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS REPLACES
 * ---------------------------------------------------------------------------
 *
 *   today (inheritance)                     here (traits)
 *   -----------------------------------     ------------------------------------
 *   class IEdge {virtual ...}               struct EdgeTrait { VTable; Ref; Mut; }
 *   class IEdgeNamed : IEdge                struct EdgeCommon {} + blanket Impl
 *   class IFieldSpan<T> : IEdgeNamed        struct FieldSpanTrait<T>  (super = Edge)
 *   class IFieldRefs<T> : IFieldSpan<T>     struct FieldRefsTrait<T>  (super = FieldSpan)
 *   class FieldRefs<T>  : IFieldRefs<T>     struct FieldRefsEdge<T> (plain struct)
 *                                             + Impl<EdgeTrait, .>       (blanket)
 *                                             + Impl<FieldSpanTrait<T>,.>
 *                                             + Impl<FieldRefsTrait<T>,.>
 *   shared_ptr<IFieldSpan<T>>               Rc<FieldSpanTrait<T>>   (= Rc<dyn FieldSpan<T>>)
 *   IFieldSpan<T>& / const IFieldSpan<T>&   FieldSpanTrait<T>::Mut / ::Ref
 *   dynamic_pointer_cast in get_ro_edge<T>  *gone*: slots are statically typed
 *   class INode {virtual ...}               struct NodeTrait + struct NodeCommon (member)
 *   class OperationSequence : INode         struct OperationSequence (plain struct)
 *
 * The key question -- "how does a FieldRefs edge decay to a FieldSpan edge?" --
 * is answered in PART 3/8: a sub-trait vtable stores a pointer to its
 * super-trait vtable, so the decay is one pointer load, statically checked,
 * and works for borrows (Ref/Mut) and owning handles (Rc) alike. It is the
 * exact analogue of Rust's `trait FieldRefs<T>: FieldSpan<T>` + trait
 * upcasting, and it replaces the `dynamic_pointer_cast` that today can only
 * fail at runtime.
 */

#include <type_traits>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// ===========================================================================
// PART 1 -- the whole trait machinery (~60 lines, written once)
// ===========================================================================
//
// Rust: `impl Trait for T { .. }`  ->  specialize `Impl<Trait, T>`.
// Rust: `dyn Trait`               ->  a (data ptr, vtable ptr) fat pointer.
// Rust: `&dyn T` / `&mut dyn T`   ->  Trait::Ref / Trait::Mut.
// Rust: `Rc<dyn Trait>`           ->  Rc<Trait>.
//
// A "trait" is a tag struct providing:
//   * struct VTable      -- the function pointer table (+ `super` if sub-trait)
//   * make<Self>()       -- builds the vtable for a concrete type (constexpr)
//   * struct Ref / Mut   -- the borrow types, i.e. the method-call sugar

/// Rust `impl Trait for Self`. Left undefined: not implementing = not usable.
template<class Trait, class Self>
struct Impl;

/// Rust `Self: Trait` bound.
template<class Trait, class Self>
concept Implements = requires { sizeof(Impl<Trait, Self>); };

/// One static vtable per (trait, concrete type) pair. Emitted at compile time,
/// lives in .rodata, exactly like a Rust vtable.
template<class Trait, class Self>
inline constexpr typename Trait::VTable vtable_of = Trait::template make<Self>();

/// Walk the `super` chain of a vtable: `&dyn Sub` -> `&dyn Super`.
/// Fully resolved at compile time; costs 0..N pointer loads, N = depth.
template<class Target, class VT>
constexpr const typename Target::VTable *vt_upcast(const VT *vt) {
    if constexpr (std::is_same_v<VT, typename Target::VTable>) {
        return vt;
    } else {
        return vt_upcast<Target>(vt->super); // `super` is the sub-trait's parent vtable
    }
}

template<class Trait, class VT>
concept UpcastableTo = requires(const VT *v) { vt_upcast<Trait>(v); };

/// Rust `Rc<dyn Trait>`: shared ownership of an erased value + its vtable.
template<class Trait>
struct Rc {
    std::shared_ptr<void> obj{};
    const typename Trait::VTable *vt = nullptr;

    Rc() = default;

    /// Rust unsized coercion `Rc<S> -> Rc<dyn Trait>` (S must implement Trait).
    template<class S>
        requires Implements<Trait, S>
    Rc(std::shared_ptr<S> p) : obj(std::move(p)), vt(&vtable_of<Trait, S>) {}

    /// Rust trait upcasting `Rc<dyn Sub> -> Rc<dyn Super>`. THE DECAY, owning flavour.
    template<class Sub>
        requires(!std::is_same_v<Sub, Trait> && UpcastableTo<Trait, typename Sub::VTable>)
    Rc(const Rc<Sub> &o) : obj(o.obj), vt(vt_upcast<Trait>(o.vt)) {}

    bool is_null() const { return obj == nullptr; }

    typename Trait::Ref as_ref() const { return {obj.get(), vt}; }
    typename Trait::Mut as_mut() const { return {obj.get(), vt}; }
};

/// Rust `&T -> &dyn Trait` for a concrete value.
template<class Trait, class S>
    requires Implements<Trait, S>
typename Trait::Ref as_ref(const S &s) {
    return {&s, &vtable_of<Trait, S>};
}
template<class Trait, class S>
    requires Implements<Trait, S>
typename Trait::Mut as_mut(S &s) {
    return {&s, &vtable_of<Trait, S>};
}

// ===========================================================================
// PART 2 -- data plumbing (stand-ins for the real shamrock types)
// ===========================================================================

template<class T>
using DD = std::map<u64, T>; ///< shambase::DistributedData

template<class T>
struct Span {
    T *ptr   = nullptr;
    u32 size = 0;
};

using SizeMap = DD<u32>;
template<class T>
using SpanMap = DD<Span<T>>; ///< DDPatchDataFieldSpanPointer<T>

/// shamrock::PatchDataField<T>
template<class T>
struct PatchField {
    std::vector<T> data;
};

template<class T>
using RefMap = DD<std::reference_wrapper<PatchField<T>>>; ///< DDPatchDataFieldRef<T>

inline u64 next_uuid() {
    static u64 c = 0;
    return c++;
}

// ===========================================================================
// PART 3 -- the edge traits
// ===========================================================================

// ---------------------------------------------------------------- IEdge ---
/// Root trait: what every edge can do (== IEdge + IEdgeNamed + IFreeable).
struct EdgeTrait {
    struct VTable {
        const char *(*type_name)();
        u64 (*uuid)(const void *);
        std::string (*label)(const void *);
        std::string (*tex_symbol)(const void *);
        void (*free_alloc)(void *);
    };

    /// `&dyn Edge`
    struct Ref {
        const void *self = nullptr;
        const VTable *vt = nullptr;
        const char *type_name() const { return vt->type_name(); }
        u64 uuid() const { return vt->uuid(self); }
        std::string label() const { return vt->label(self); }
        std::string tex_symbol() const { return vt->tex_symbol(self); }
    };

    /// `&mut dyn Edge`
    struct Mut {
        void *self       = nullptr;
        const VTable *vt = nullptr;
        operator Ref() const { return {self, vt}; } // &mut T -> &T
        const char *type_name() const { return vt->type_name(); }
        u64 uuid() const { return vt->uuid(self); }
        std::string label() const { return vt->label(self); }
        std::string tex_symbol() const { return vt->tex_symbol(self); }
        void free_alloc() const { vt->free_alloc(self); }
    };

    template<class S>
    static constexpr VTable make() {
        using I = Impl<EdgeTrait, S>;
        return VTable{
            []() {
                return typeid(S).name();
            },
            [](const void *s) {
                return I::uuid(*static_cast<const S *>(s));
            },
            [](const void *s) {
                return I::label(*static_cast<const S *>(s));
            },
            [](const void *s) {
                return I::tex_symbol(*static_cast<const S *>(s));
            },
            [](void *s) {
                I::free_alloc(*static_cast<S *>(s));
            }};
    }
};

using EdgeRef = EdgeTrait::Ref;
using EdgeRc  = Rc<EdgeTrait>;

/// Replaces the *state* that IEdgeNamed/WithUUID injected by inheritance:
/// now a member, not a base.
struct EdgeCommon {
    u64 uuid = next_uuid();
    std::string name;
    std::string tex;
};

template<class S>
concept HasEdgeCommon = requires(S &s) {
    { s.common } -> std::convertible_to<EdgeCommon &>;
};

/// Rust *blanket impl*: `impl<S: HasEdgeCommon> Edge for S`.
/// This single specialization is what IEdgeNamed used to be -- except no edge
/// type has to inherit from anything to get it.
template<HasEdgeCommon S>
struct Impl<EdgeTrait, S> {
    static u64 uuid(const S &s) { return s.common.uuid; }
    static std::string label(const S &s) { return s.common.name; }
    static std::string tex_symbol(const S &s) { return "{" + s.common.tex + "}"; }
    static void free_alloc(S &s) {
        // Rust: a default method body, overridden if the type provides one.
        if constexpr (requires { s.free_alloc(); }) {
            s.free_alloc();
        }
    }
};

// ----------------------------------------------------------- IFieldSpan ---
/// `trait FieldSpan<T>: Edge`
template<class T>
struct FieldSpanTrait {
    struct VTable {
        const EdgeTrait::VTable *super; // <<< super-trait vtable == the decay
        const SpanMap<T> &(*spans)(const void *);
        SpanMap<T> &(*spans_mut)(void *);
        void (*check_sizes)(const void *, const SizeMap &);
        void (*ensure_sizes)(void *, const SizeMap &);
    };

    struct Ref {
        const void *self = nullptr;
        const VTable *vt = nullptr;
        operator EdgeRef() const { return {self, vt->super}; } // decay to &dyn Edge
        const SpanMap<T> &spans() const { return vt->spans(self); }
        void check_sizes(const SizeMap &s) const { vt->check_sizes(self, s); }
    };

    struct Mut {
        void *self       = nullptr;
        const VTable *vt = nullptr;
        operator Ref() const { return {self, vt}; }
        operator EdgeRef() const { return {self, vt->super}; }
        const SpanMap<T> &spans() const { return vt->spans(self); }
        SpanMap<T> &spans_mut() const { return vt->spans_mut(self); }
        void check_sizes(const SizeMap &s) const { vt->check_sizes(self, s); }
        void ensure_sizes(const SizeMap &s) const { vt->ensure_sizes(self, s); }
    };

    template<class S>
    static constexpr VTable make() {
        using I = Impl<FieldSpanTrait<T>, S>;
        return VTable{
            &vtable_of<EdgeTrait, S>, // same S -> the chain never loses the concrete type
            [](const void *s) -> const SpanMap<T> & {
                return I::spans(*static_cast<const S *>(s));
            },
            [](void *s) -> SpanMap<T> & {
                return I::spans_mut(*static_cast<S *>(s));
            },
            [](const void *s, const SizeMap &sz) {
                I::check_sizes(*static_cast<const S *>(s), sz);
            },
            [](void *s, const SizeMap &sz) {
                I::ensure_sizes(*static_cast<S *>(s), sz);
            }};
    }
};

// ----------------------------------------------------------- IFieldRefs ---
/// `trait FieldRefs<T>: FieldSpan<T>`
template<class T>
struct FieldRefsTrait {
    struct VTable {
        const typename FieldSpanTrait<T>::VTable *super; // <<< THE DECAY EDGE
        const RefMap<T> &(*refs)(const void *);
        RefMap<T> &(*refs_mut)(void *);
    };

    struct Ref {
        const void *self = nullptr;
        const VTable *vt = nullptr;
        /// `&dyn FieldRefs<T>` -> `&dyn FieldSpan<T>`: ONE pointer load, no RTTI.
        operator typename FieldSpanTrait<T>::Ref() const { return {self, vt->super}; }
        operator EdgeRef() const { return {self, vt->super->super}; }
        const RefMap<T> &refs() const { return vt->refs(self); }
        PatchField<T> &field(u64 id) const { return refs().at(id).get(); }
    };

    struct Mut {
        void *self       = nullptr;
        const VTable *vt = nullptr;
        operator Ref() const { return {self, vt}; }
        operator typename FieldSpanTrait<T>::Mut() const { return {self, vt->super}; }
        operator typename FieldSpanTrait<T>::Ref() const { return {self, vt->super}; }
        operator EdgeRef() const { return {self, vt->super->super}; }
        const RefMap<T> &refs() const { return vt->refs(self); }
        RefMap<T> &refs_mut() const { return vt->refs_mut(self); }
    };

    template<class S>
    static constexpr VTable make() {
        using I = Impl<FieldRefsTrait<T>, S>;
        return VTable{
            &vtable_of<FieldSpanTrait<T>, S>,
            [](const void *s) -> const RefMap<T> & {
                return I::refs(*static_cast<const S *>(s));
            },
            [](void *s) -> RefMap<T> & {
                return I::refs_mut(*static_cast<S *>(s));
            }};
    }
};

// ===========================================================================
// PART 4 -- concrete edges: plain structs, zero base classes
// ===========================================================================

/// == solvergraph::IDataEdge<T>. Note: no trait at all, it is used as a
/// statically typed slot (see ForwardEuler::dt) -- erasure is opt-in.
template<class T>
struct DataEdge {
    EdgeCommon common;
    T data{};
    DataEdge(std::string name, std::string tex) : common{next_uuid(), name, tex} {}
    void free_alloc() { data = {}; }
};

/// == solvergraph::FieldSpan<T>
template<class T>
struct FieldSpanEdge {
    EdgeCommon common;
    SpanMap<T> spans;
    FieldSpanEdge(std::string name, std::string tex) : common{next_uuid(), name, tex} {}
    void free_alloc() { spans.clear(); }
};

template<class T>
struct Impl<FieldSpanTrait<T>, FieldSpanEdge<T>> {
    static const SpanMap<T> &spans(const FieldSpanEdge<T> &s) { return s.spans; }
    static SpanMap<T> &spans_mut(FieldSpanEdge<T> &s) { return s.spans; }
    static void check_sizes(const FieldSpanEdge<T> &s, const SizeMap &sizes) {
        for (auto &[id, n] : sizes) {
            auto it = s.spans.find(id);
            if (it == s.spans.end() || it->second.size != n) {
                throw std::runtime_error(
                    "size mismatch on span edge '" + s.common.name + "' at patch "
                    + std::to_string(id));
            }
        }
    }
    /// a span edge cannot grow: it does not own the memory
    static void ensure_sizes(FieldSpanEdge<T> &s, const SizeMap &sizes) { check_sizes(s, sizes); }
};

/// == solvergraph::FieldRefs<T>: owns references to the patch fields and keeps
/// a derived span view in sync.
template<class T>
struct FieldRefsEdge {
    EdgeCommon common;
    RefMap<T> field_refs;
    SpanMap<T> spans;

    FieldRefsEdge(std::string name, std::string tex) : common{next_uuid(), name, tex} {}

    void sync_spans() {
        spans.clear();
        for (auto &[id, ref] : field_refs) {
            auto &v   = ref.get().data;
            spans[id] = Span<T>{v.data(), static_cast<u32>(v.size())};
        }
    }
    void set_refs(RefMap<T> r) {
        field_refs = std::move(r);
        sync_spans();
    }
    void free_alloc() {
        field_refs.clear();
        spans.clear();
    }
};

template<class T>
struct Impl<FieldSpanTrait<T>, FieldRefsEdge<T>> {
    static const SpanMap<T> &spans(const FieldRefsEdge<T> &s) { return s.spans; }
    static SpanMap<T> &spans_mut(FieldRefsEdge<T> &s) { return s.spans; }
    static void check_sizes(const FieldRefsEdge<T> &s, const SizeMap &sizes) {
        for (auto &[id, n] : sizes) {
            if (s.field_refs.find(id) == s.field_refs.end()) {
                throw std::runtime_error("missing field ref at patch " + std::to_string(id));
            }
        }
    }
    /// unlike FieldSpanEdge, this one OWNS the fields: it can resize + resync.
    /// The node calling ensure_sizes() through a decayed FieldSpan view gets
    /// this behaviour -- that is the dynamic dispatch we kept.
    static void ensure_sizes(FieldRefsEdge<T> &s, const SizeMap &sizes) {
        check_sizes(s, sizes);
        bool dirty = false;
        for (auto &[id, n] : sizes) {
            auto &v = s.field_refs.at(id).get().data;
            if (v.size() != n) {
                v.resize(n);
                dirty = true;
            }
        }
        if (dirty) {
            s.sync_spans();
        }
    }
};

template<class T>
struct Impl<FieldRefsTrait<T>, FieldRefsEdge<T>> {
    static const RefMap<T> &refs(const FieldRefsEdge<T> &s) { return s.field_refs; }
    static RefMap<T> &refs_mut(FieldRefsEdge<T> &s) { return s.field_refs; }
};

// ===========================================================================
// PART 5 -- the node trait (== INode)
// ===========================================================================

/// Replaces the state INode injected through inheritance (uuid + edge lists).
struct NodeCommon {
    u64 uuid = next_uuid();
};

struct NodeTrait {
    struct VTable {
        const char *(*type_name)();
        const NodeCommon &(*common)(const void *);
        void (*evaluate)(void *);
        std::string (*label)(const void *);
        std::string (*tex)(const void *);
        std::vector<EdgeRef> (*ro_edges)(const void *);
        std::vector<EdgeRef> (*rw_edges)(const void *);
        // the three below have default bodies, overridable per impl
        std::string (*dot_partial)(const void *);
        std::string (*dot_start)(const void *);
        std::string (*dot_end)(const void *);
    };

    struct Ref {
        const void *self = nullptr;
        const VTable *vt = nullptr;
        const char *type_name() const { return vt->type_name(); }
        u64 uuid() const { return vt->common(self).uuid; }
        std::string label() const { return vt->label(self); }
        std::string tex() const { return vt->tex(self); }
        std::vector<EdgeRef> ro_edges() const { return vt->ro_edges(self); }
        std::vector<EdgeRef> rw_edges() const { return vt->rw_edges(self); }
        std::string dot_partial() const { return vt->dot_partial(self); }
        std::string dot_start() const { return vt->dot_start(self); }
        std::string dot_end() const { return vt->dot_end(self); }
        std::string print_node_info() const;
    };

    struct Mut {
        void *self       = nullptr;
        const VTable *vt = nullptr;
        operator Ref() const { return {self, vt}; }
        void evaluate() const { vt->evaluate(self); }
        std::string label() const { return vt->label(self); }
        std::string dot_partial() const { return vt->dot_partial(self); }
    };

    /// default `_impl_get_dot_graph_partial`, shared by every node that does
    /// not override it (was: a virtual with a body in the base class).
    template<class S>
    static std::string default_dot_partial(const S &s) {
        Ref r = as_ref<NodeTrait>(s);
        std::stringstream ss;
        ss << "n_" << r.uuid() << " [label=\"" << r.label() << "\"];\n";
        for (EdgeRef e : r.ro_edges()) {
            ss << "e_" << e.uuid() << " -> n_" << r.uuid() << " [style=\"dashed\", color=green];\n";
            ss << "e_" << e.uuid() << " [label=\"" << e.label() << "\",shape=rect];\n";
        }
        for (EdgeRef e : r.rw_edges()) {
            ss << "n_" << r.uuid() << " -> e_" << e.uuid() << " [style=\"dashed\", color=red];\n";
            ss << "e_" << e.uuid() << " [label=\"" << e.label() << "\",shape=rect];\n";
        }
        return ss.str();
    }

    template<class S>
    static constexpr VTable make() {
        using I = Impl<NodeTrait, S>;
        VTable vt{};
        vt.type_name = []() {
            return typeid(S).name();
        };
        vt.common = [](const void *s) -> const NodeCommon & {
            return I::common(*static_cast<const S *>(s));
        };
        vt.evaluate = [](void *s) {
            I::evaluate(*static_cast<S *>(s));
        };
        vt.label = [](const void *s) {
            return I::label(*static_cast<const S *>(s));
        };
        vt.tex = [](const void *s) {
            return I::tex(*static_cast<const S *>(s));
        };
        // -- default method bodies, overridden when the impl provides one -----
        if constexpr (requires { I::ro_edges; }) {
            vt.ro_edges = [](const void *s) {
                return I::ro_edges(*static_cast<const S *>(s));
            };
        } else {
            vt.ro_edges = [](const void *) {
                return std::vector<EdgeRef>{};
            };
        }
        if constexpr (requires { I::rw_edges; }) {
            vt.rw_edges = [](const void *s) {
                return I::rw_edges(*static_cast<const S *>(s));
            };
        } else {
            vt.rw_edges = [](const void *) {
                return std::vector<EdgeRef>{};
            };
        }
        if constexpr (requires { I::dot_partial; }) {
            vt.dot_partial = [](const void *s) {
                return I::dot_partial(*static_cast<const S *>(s));
            };
        } else {
            vt.dot_partial = [](const void *s) {
                return default_dot_partial(*static_cast<const S *>(s));
            };
        }
        if constexpr (requires { I::dot_start; }) {
            vt.dot_start = [](const void *s) {
                return I::dot_start(*static_cast<const S *>(s));
            };
        } else {
            vt.dot_start = [](const void *s) {
                return "n_" + std::to_string(I::common(*static_cast<const S *>(s)).uuid);
            };
        }
        if constexpr (requires { I::dot_end; }) {
            vt.dot_end = [](const void *s) {
                return I::dot_end(*static_cast<const S *>(s));
            };
        } else {
            vt.dot_end = [](const void *s) {
                return "n_" + std::to_string(I::common(*static_cast<const S *>(s)).uuid);
            };
        }
        return vt;
    }
};

using NodeRef = NodeTrait::Ref;
using NodeRc  = Rc<NodeTrait>;

inline std::string NodeTrait::Ref::print_node_info() const {
    std::stringstream ss;
    ss << "Node info :\n";
    ss << " - Node type : " << type_name() << "\n";
    ss << " - Node UUID : " << uuid() << "\n";
    ss << " - Node label : " << label() << "\n";
    auto dump = [&](const char *title, const std::vector<EdgeRef> &edges) {
        ss << " - " << title << ": " << edges.size() << "\n";
        for (EdgeRef e : edges) {
            ss << "     - uuid = " << e.uuid() << ", label = " << e.label()
               << ", type = " << e.type_name() << "\n";
        }
    };
    dump("Node Read Only edges", ro_edges());
    dump("Node Read Write edges", rw_edges());
    return ss.str();
}

// ===========================================================================
// PART 6 -- a real node: ForwardEuler
// ===========================================================================
//
// Compare with shammodels::common::modules::ForwardEuler:
//   X_RO(IDataEdge<Tscal>, dt)          -> shared_ptr<DataEdge<Tscal>>  (typed)
//   X_RO(IFieldSpan<T>, time_derivative)-> Rc<FieldSpanTrait<T>>        (dyn)
//   X_RW(IFieldSpan<T>, field)          -> Rc<FieldSpanTrait<T>>        (dyn)
//
// The slots are typed, so `get_edges()` needs NO dynamic_pointer_cast: a wrong
// edge type is now a compile error at the set_edges() call site instead of a
// throw inside evaluate().

template<class T>
struct ForwardEuler {
    NodeCommon common;
    u32 nvar = 1;

    // ---- slots (what `__internal_set_ro_edges` erased into a vector) -------
    std::shared_ptr<DataEdge<T>> dt;       // ro, statically typed
    Rc<FieldSpanTrait<T>> time_derivative; // ro, Rc<dyn FieldSpan<T>>
    Rc<FieldSpanTrait<T>> field;           // rw, Rc<dyn FieldSpan<T>>

    explicit ForwardEuler(u32 nvar = 1) : nvar(nvar) {}

    /// == the EXPAND_NODE_EDGES-generated `Edges` struct: RO slots come out as
    /// `Ref` (read only *by type*), RW slots as `Mut`.
    struct Edges {
        const T &dt;
        typename FieldSpanTrait<T>::Ref time_derivative;
        typename FieldSpanTrait<T>::Mut field;
    };

    void set_edges(
        std::shared_ptr<DataEdge<T>> dt_,
        Rc<FieldSpanTrait<T>> time_derivative_, // any edge implementing FieldSpan<T> decays here
        Rc<FieldSpanTrait<T>> field_) {
        dt              = std::move(dt_);
        time_derivative = std::move(time_derivative_);
        field           = std::move(field_);
    }

    Edges get_edges() const { return Edges{dt->data, time_derivative.as_ref(), field.as_mut()}; }
};

template<class T>
struct Impl<NodeTrait, ForwardEuler<T>> {
    static const NodeCommon &common(const ForwardEuler<T> &s) { return s.common; }
    static std::string label(const ForwardEuler<T> &) { return "ForwardEuler"; }
    static std::string tex(const ForwardEuler<T> &) {
        return "$f \\mathrel{+}= \\Delta t\\,\\dot f$";
    }

    static std::vector<EdgeRef> ro_edges(const ForwardEuler<T> &s) {
        return {as_ref<EdgeTrait>(*s.dt), s.time_derivative.as_ref()};
    }
    static std::vector<EdgeRef> rw_edges(const ForwardEuler<T> &s) { return {s.field.as_ref()}; }

    static void evaluate(ForwardEuler<T> &s) {
        auto edges = s.get_edges();

        SizeMap sizes;
        for (auto &[id, sp] : edges.time_derivative.spans()) {
            sizes[id] = sp.size;
        }
        edges.field.ensure_sizes(sizes); // dispatches to FieldRefsEdge::ensure_sizes if that is
                                         // the concrete type behind the decayed view

        const T dt = edges.dt;
        for (auto &[id, dst] : edges.field.spans_mut()) {
            const Span<T> &src = edges.time_derivative.spans().at(id);
            for (u32 i = 0; i < dst.size * s.nvar; i++) {
                dst.ptr[i] = dst.ptr[i] + dt * src.ptr[i];
            }
        }
    }
};

// ===========================================================================
// PART 7 -- OperationSequence, without inheriting from INode
// ===========================================================================

struct OperationSequence {
    NodeCommon common;
    std::string name;
    std::vector<NodeRc> nodes;

    OperationSequence(std::string name, std::vector<NodeRc> nodes)
        : name(std::move(name)), nodes(std::move(nodes)) {
        if (this->nodes.empty()) {
            throw std::invalid_argument("OperationSequence must have at least one node");
        }
    }
};

template<>
struct Impl<NodeTrait, OperationSequence> {
    static const NodeCommon &common(const OperationSequence &s) { return s.common; }
    static std::string label(const OperationSequence &s) { return s.name; }

    static void evaluate(OperationSequence &s) {
        for (NodeRc &n : s.nodes) {
            n.as_mut().evaluate();
        }
    }

    static std::string tex(const OperationSequence &s) {
        std::stringstream ss;
        ss << "Start : " << s.name << "\n";
        for (const NodeRc &n : s.nodes) {
            ss << n.as_ref().tex() << "\n";
        }
        ss << "End : " << s.name << "\n";
        return ss.str();
    }

    // --- these three OVERRIDE the trait's default bodies -------------------
    static std::string dot_partial(const OperationSequence &s) {
        std::stringstream ss;
        ss << "subgraph cluster_" << s.common.uuid << " {\n";
        for (const NodeRc &n : s.nodes) {
            ss << n.as_ref().dot_partial();
        }
        for (std::size_t i = 0; i + 1 < s.nodes.size(); i++) {
            ss << s.nodes[i].as_ref().dot_end() << " -> " << s.nodes[i + 1].as_ref().dot_start()
               << " [weight=3];\n";
        }
        ss << "label = \"" << s.name << "\";\n}\n";
        return ss.str();
    }
    static std::string dot_start(const OperationSequence &s) {
        return s.nodes.front().as_ref().dot_start();
    }
    static std::string dot_end(const OperationSequence &s) {
        return s.nodes.back().as_ref().dot_end();
    }
};

// ===========================================================================
// PART 8 -- the decay, demonstrated
// ===========================================================================

static void decay_demo() {
    std::cout << "== decay demo ==\n";

    PatchField<double> f0{{1., 2., 3.}};

    auto refs_edge = std::make_shared<FieldRefsEdge<double>>("rho", "\\rho");
    refs_edge->set_refs({{0, std::ref(f0)}});

    // (a) concrete &FieldRefsEdge<double> -> &dyn FieldSpan<double>
    //     (Rust: `&r as &dyn FieldSpan<f64>`, resolved statically)
    typename FieldSpanTrait<double>::Ref span_view = as_ref<FieldSpanTrait<double>>(*refs_edge);
    std::cout << "  a) &FieldRefsEdge -> &dyn FieldSpan : n=" << span_view.spans().at(0).size
              << "\n";

    // (b) &dyn FieldRefs<double> -> &dyn FieldSpan<double>  (trait upcast,
    //     one load of vt->super; this is what `IFieldRefs<T>&` -> `IFieldSpan<T>&`
    //     is today, minus the vtable-offset machinery)
    typename FieldRefsTrait<double>::Ref refs_view = as_ref<FieldRefsTrait<double>>(*refs_edge);
    typename FieldSpanTrait<double>::Ref decayed   = refs_view; // implicit
    EdgeRef as_plain_edge                          = refs_view; // two levels up, still implicit
    std::cout << "  b) &dyn FieldRefs -> &dyn FieldSpan : n=" << decayed.spans().at(0).size
              << ", label=" << as_plain_edge.label() << "\n";

    // (c) owning: Rc<dyn FieldRefs<T>> -> Rc<dyn FieldSpan<T>> -> Rc<dyn Edge>
    //     (Rust: `Rc<dyn FieldRefs<f64>> as Rc<dyn FieldSpan<f64>>`)
    Rc<FieldRefsTrait<double>> rc_refs = refs_edge;
    Rc<FieldSpanTrait<double>> rc_span = rc_refs; // decay, refcount shared
    EdgeRc rc_edge                     = rc_span; // decay again
    std::cout << "  c) Rc<dyn FieldRefs> -> Rc<dyn FieldSpan> -> Rc<dyn Edge> : uuid="
              << rc_edge.as_ref().uuid() << ", tex=" << rc_edge.as_ref().tex_symbol() << "\n";

    // (d) shared_ptr<FieldRefsEdge<T>> -> Rc<dyn FieldSpan<T>> in one step,
    //     which is exactly what happens at a set_edges() call site.
    Rc<FieldSpanTrait<double>> direct = refs_edge;
    std::cout << "  d) shared_ptr<FieldRefsEdge> -> Rc<dyn FieldSpan> : ok\n";

    // (e) what does NOT compile (uncomment to see):
    //   Rc<FieldRefsTrait<double>> bad = std::make_shared<FieldSpanEdge<double>>("v", "v");
    //     -> no Impl<FieldRefsTrait<double>, FieldSpanEdge<double>>, so the
    //        Rc constructor is not viable. Today this is a dynamic_pointer_cast
    //        that returns null and throws at runtime.
    //   decayed.spans_mut();
    //     -> `Ref` has no mutating method: read-only is enforced by the type.
    (void) direct;
    std::cout << "\n";
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    decay_demo();

    std::cout << "== graph demo ==\n";

    // --- the data --------------------------------------------------------
    PatchField<double> vel_p0{{0., 0., 0., 0.}};
    PatchField<double> vel_p1{{1., 1.}};
    PatchField<double> acc_p0{{1., 2., 3., 4.}};
    PatchField<double> acc_p1{{-1., -2.}};

    // --- the edges -------------------------------------------------------
    auto dt  = std::make_shared<DataEdge<double>>("dt", "\\Delta t");
    dt->data = 0.5;

    auto acc = std::make_shared<FieldSpanEdge<double>>("acc", "a");
    acc->spans
        = {{0, Span<double>{acc_p0.data.data(), 4}}, {1, Span<double>{acc_p1.data.data(), 2}}};

    auto vel = std::make_shared<FieldRefsEdge<double>>("vel", "v");
    vel->set_refs({{0, std::ref(vel_p0)}, {1, std::ref(vel_p1)}});

    // --- the nodes -------------------------------------------------------
    auto euler1 = std::make_shared<ForwardEuler<double>>();
    // `acc` is a FieldSpanEdge, `vel` is a FieldRefsEdge: both decay into
    // Rc<dyn FieldSpan<double>> right here, at the call site.
    euler1->set_edges(dt, acc, vel);

    auto euler2 = std::make_shared<ForwardEuler<double>>();
    euler2->set_edges(dt, acc, vel);

    auto seq = std::make_shared<OperationSequence>(
        "kick x2", std::vector<NodeRc>{NodeRc(euler1), NodeRc(euler2)});

    // --- run -------------------------------------------------------------
    NodeRc graph = seq;
    graph.as_mut().evaluate();

    std::cout << "vel patch0 =";
    for (double v : vel_p0.data) {
        std::cout << " " << v;
    }
    std::cout << "   (expected 1 2 3 4)\n";
    std::cout << "vel patch1 =";
    for (double v : vel_p1.data) {
        std::cout << " " << v;
    }
    std::cout << "   (expected 0 -1)\n\n";

    std::cout << as_ref<NodeTrait>(*euler1).print_node_info() << "\n";
    std::cout << "digraph G {\n" << graph.as_ref().dot_partial() << "}\n";

    // free_alloc through the erased edge handle (== IFreeable)
    EdgeRc(Rc<FieldSpanTrait<double>>(vel)).as_mut().free_alloc();
    std::cout << "\nafter free_alloc: vel spans = " << vel->spans.size() << "\n";

    std::cout << "\nsizeof(Rc<dyn FieldSpan<double>>) = " << sizeof(Rc<FieldSpanTrait<double>>)
              << " (shared_ptr + vtable ptr)\n";
    std::cout << "sizeof(FieldSpanTrait<double>::Ref) = "
              << sizeof(typename FieldSpanTrait<double>::Ref) << " (fat pointer)\n";
    return 0;
}
