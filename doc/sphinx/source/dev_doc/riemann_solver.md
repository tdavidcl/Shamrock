# Riemann solver implementation notes

This page documents implementation choices made in `shammath`'s Riemann solvers (Rusanov, HLL,
HLLC, dust HLL, Huang-Bai).

## Axis permutation vs projection

Every Riemann solver is defined once as an `_n` variant that takes the face's unit normal `n`
directly:

```cpp
template<class Tprim>
inline constexpr auto riemann_solver_flux_n(
    Tprim primL, Tprim primR, typename Tprim::Tscal gamma, typename Tprim::Tvec n) {
    // ... flux computation using n[0], n[1], n[2] directly ...
}
```

Until recently, each solver also shipped six `_x`/`_y`/`_z`/`_mx`/`_my`/`_mz` wrappers that got to
the same result by **permuting** components to the `+x` axis, calling the `_x` solver, then
permuting the result back — instead of **projecting** through `_n` with the axis's unit vector
directly. E.g. the `-z` wrapper:

```cpp
// axis permutation: rotate the inputs to +x, solve there, rotate the result back
template<class Tprim>
inline constexpr auto riemann_solver_flux_mz(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
    return invert_axis(
        riemann_solver_flux_z(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
}
```

`riemann_solver_flux_mz(pL, pR, gamma)` and `riemann_solver_flux_n(pL, pR, gamma, {0, 0, -1})`
are mathematically equivalent, but they are not equivalent *as generated code*. All six wrappers
have been removed from every solver; callers such as `ComputeFluxUtilities.hpp` now build the
direction's unit vector once (from the `Direction` enum) and call `_n` directly.

### A worked example

{download}`riemann_solver_axis_dispatch_godbolt.cpp` is a minimal, dependency-free repro (no
SYCL, just a plain `Vec3`) isolating exactly this comparison for one solver, with two entry
points:

```cpp
Cons via_mz_dispatch(Prim pL, Prim pR) {
    return riemann_solver_flux_mz(pL, pR);
}

Cons via_flux_n(Prim pL, Prim pR) {
    return riemann_solver_flux_n(pL, pR, Vec3{0, 0, -1});
}
```

Compiled on [Compiler Explorer](https://godbolt.org) with x86-64 clang at `-O2`, both fully
inlined into a single leaf function, the two disassemble to:

```
.LCPI0_0:
        .quad   0x8000000000000000
        .quad   0x8000000000000000
via_mz_dispatch(DustPrimState<Vec3>, DustPrimState<Vec3>):
        mov     rax, rdi
        movapd  xmm1, xmmword ptr [rsp + 8]
        movsd   xmm5, qword ptr [rsp + 32]
        movsd   xmm4, qword ptr [rsp + 64]
        movsd   xmm0, qword ptr [rsp + 24]
        movapd  xmm3, xmmword ptr [rip + .LCPI0_0]
        xorpd   xmm0, xmm3
        movsd   xmm6, qword ptr [rsp + 56]
        xorpd   xmm6, xmm3
        movapd  xmm2, xmm5
        unpcklpd        xmm2, xmm4
        xorpd   xmm2, xmm3
        movhpd  xmm1, qword ptr [rsp + 40]
        mulpd   xmm1, xmm2
        movsd   xmm3, qword ptr [rsp + 16]
        mulsd   xmm3, xmm1
        shufpd  xmm0, xmm2, 2
        mulpd   xmm0, xmm1
        unpcklpd        xmm2, xmm6
        mulpd   xmm2, xmm1
        movapd  xmm6, xmm1
        unpckhpd        xmm6, xmm1
        movsd   xmm7, qword ptr [rsp + 48]
        mulsd   xmm7, xmm6
        xorpd   xmm8, xmm8
        ucomisd xmm8, xmm5
        jbe     .LBB0_4
        ucomisd xmm4, xmm8
        jae     .LBB0_4
        unpcklpd        xmm0, xmm2
        movapd  xmm8, xmm1
        movapd  xmm4, xmm3
.LBB0_3:
        movapd  xmm5, xmm0
        jmp     .LBB0_7
.LBB0_4:
        ucomisd xmm5, xmm8
        jbe     .LBB0_8
        ucomisd xmm4, xmm8
        jbe     .LBB0_8
        unpckhpd        xmm2, xmm0
        movapd  xmm8, xmm6
        movapd  xmm4, xmm7
        movapd  xmm5, xmm2
.LBB0_7:
        xorpd   xmm5, xmmword ptr [rip + .LCPI0_0]
        movsd   qword ptr [rax], xmm8
        movsd   qword ptr [rax + 8], xmm4
        movupd  xmmword ptr [rax + 16], xmm5
        ret
.LBB0_8:
        ucomisd xmm5, xmm8
        seta    cl
        ucomisd xmm4, xmm8
        setb    dl
        ucomisd xmm5, xmm8
        xorpd   xmm5, xmm5
        jae     .LBB0_13
        and     cl, dl
        jne     .LBB0_13
        xorpd   xmm9, xmm9
        ucomisd xmm4, xmm9
        xorpd   xmm4, xmm4
        jbe     .LBB0_7
        shufpd  xmm2, xmm2, 1
        addsd   xmm6, xmm1
        addpd   xmm0, xmm2
        addsd   xmm7, xmm3
        movapd  xmm8, xmm6
        movapd  xmm4, xmm7
        jmp     .LBB0_3
.LBB0_13:
        xorpd   xmm4, xmm4
        jmp     .LBB0_7

.LCPI1_0:
        .quad   0x8000000000000000
        .quad   0x8000000000000000
via_flux_n(DustPrimState<Vec3>, DustPrimState<Vec3>):
        mov     rax, rdi
        movsd   xmm5, qword ptr [rsp + 32]
        movsd   xmm2, qword ptr [rsp + 64]
        movapd  xmm3, xmmword ptr [rip + .LCPI1_0]
        movsd   xmm0, qword ptr [rsp + 8]
        xorpd   xmm0, xmm3
        movsd   xmm1, qword ptr [rsp + 40]
        xorpd   xmm1, xmm3
        mulsd   xmm0, xmm5
        movsd   xmm7, qword ptr [rsp + 16]
        mulsd   xmm7, xmm0
        movapd  xmm3, xmm0
        unpcklpd        xmm3, xmm0
        mulpd   xmm3, xmmword ptr [rsp + 24]
        mulsd   xmm1, xmm2
        movapd  xmm4, xmm1
        unpcklpd        xmm4, xmm1
        mulpd   xmm4, xmmword ptr [rsp + 56]
        xorpd   xmm6, xmm6
        ucomisd xmm6, xmm5
        unpcklpd        xmm0, xmm7
        jbe     .LBB1_4
        ucomisd xmm2, xmm6
        jae     .LBB1_4
        movapd  xmm5, xmm0
        movapd  xmm2, xmm3
.LBB1_3:
        movupd  xmmword ptr [rax], xmm5
        movupd  xmmword ptr [rax + 16], xmm2
        ret
.LBB1_4:
        lea     rcx, [rsp + 40]
        movapd  xmm7, xmm1
        mulsd   xmm7, qword ptr [rcx + 8]
        ucomisd xmm5, xmm6
        unpcklpd        xmm1, xmm7
        jbe     .LBB1_6
        ucomisd xmm2, xmm6
        ja      .LBB1_10
.LBB1_6:
        ucomisd xmm5, xmm6
        seta    cl
        ucomisd xmm2, xmm6
        setb    dl
        ucomisd xmm5, xmm6
        xorpd   xmm5, xmm5
        jae     .LBB1_12
        and     cl, dl
        jne     .LBB1_12
        ucomisd xmm2, xmm6
        xorpd   xmm2, xmm2
        jbe     .LBB1_3
        addpd   xmm1, xmm0
        addpd   xmm4, xmm3
.LBB1_10:
        movapd  xmm5, xmm1
        movapd  xmm2, xmm4
        movupd  xmmword ptr [rax], xmm5
        movupd  xmmword ptr [rax + 16], xmm2
        ret
.LBB1_12:
        xorpd   xmm2, xmm2
        movupd  xmmword ptr [rax], xmm5
        movupd  xmmword ptr [rax + 16], xmm2
        ret
```

Both keep the same control-flow shape (the solver's internal branch tree survives inlining
unchanged), but `via_mz_dispatch` does strictly more work for the same result:

- **More sign flips.** `via_flux_n` needs 2 real `xorpd`s (negating the `z` component once per
  side, since `n = {0, 0, -1}`). `via_mz_dispatch` needs 4 — two from rotating the inputs in
  (`prim_invert_axis` + `prim_z_to_x`) that don't fully cancel against the two undoing the
  rotation on the way out (`x_to_z` + `invert_axis`); one extra negation survives all the way to
  the final store with no counterpart in the direct-`n` version.
- **Lane shuffling.** `via_mz_dispatch` uses `unpcklpd`/`unpckhpd`/`shufpd`/`movhpd` to move
  vector components between lanes, a byproduct of routing everything through the `+x` axis.
  `via_flux_n` never permutes lanes: with `n_x = n_y = 0`, the relevant component is used in
  place.
- **More live registers and instructions for an identical result** (`via_mz_dispatch` reaches
  `xmm9`, `via_flux_n` stops at `xmm7`).

The compiler eliminates the literal-zero multiplies coming from `{1, 0, 0}` in both cases, but it
does not fully cancel the round trip's redundant sign flips and lane permutes. Calling `_n`
directly with the target axis's unit vector is not just cleaner source — it is strictly cheaper
codegen at `-O2`.
