# Contribution rules (draft)

> Temporary working file. These rules will be folded into the real
> `CONTRIBUTING.md` once it is written. Not meant to be committed as is.

## 1. A disabled feature must not change the existing code path

Adding a feature must leave the code path used when that feature is
**disabled** exactly as it was before. With the feature off, the code
that runs must be the same code as before the PR, not "the same code
plus a branch that is not taken".

In practice, a new option (boolean, enum, mode, ...) **selects which
kernel or code path runs**. It is **not** passed into an existing kernel
and tested there.

Bad, because the flag is evaluated inside the existing kernel:

```cpp
sham::kernel_call(q, sham::MultiRef{...}, sham::MultiRef{...}, n,
    [=](u32 i, ...) {
        // existing computation
        if (use_new_feature) {
            // new computation
        }
    });
```

Even when `use_new_feature` is false, this changes the existing kernel:
more registers, kernel arguments and branches, and different compiler
optimizations. It can slow down the disabled path and change its last
bits.

Good, because the choice is made on the host and the existing kernel is
untouched:

```cpp
if (use_new_feature) {
    run_kernel_with_feature(...); // new kernel
} else {
    run_existing_kernel(...);     // unchanged
}
```

A compile-time switch (`template<bool with_feature>` with `if constexpr`)
is also acceptable, as long as the `false` instantiation compiles to the
same kernel as before.

This rule is not limited to kernels. When the feature is disabled:

- no extra fields are allocated or carried in the patch data;
- no extra ghost exchanges or MPI communications run;
- no extra solver graph nodes are executed;
- no extra memory is used.

Why: it means "feature off" behaves exactly like the previous version,
both in results and in performance. It also makes this property easy to
check in review: the existing path either appears in the diff or it
doesn't.

## 2. Determinism

Shamrock is bitwise run-to-run deterministic: the same binary, on the
same hardware, with the same inputs, must give bit-identical outputs
from one run to the next. A change that breaks this is a regression,
even if the results stay within floating-point tolerance or are still
"physically correct". Determinism is never traded for speed.

Before touching kernels, reductions, communications or the scheduler,
read `doc/sphinx/source/dev_doc/determinism.md`. It covers what breaks
determinism (floating-point atomics, scheduling-dependent reductions,
arrival-order MPI processing, unordered containers, timing-based
decisions, ...), the patterns that keep it, and how to check it.

New code paths must be deterministic too, including a new feature when
it is enabled.

## 3. No regression, including performance

A PR must not regress existing functionality, either in correctness or
in performance:

- existing tests keep passing;
- with any new feature disabled, outputs are bit-identical to the
  previous version on the same machine. This follows from rules 1
  and 2;
- no measurable slowdown on existing paths. A change that deliberately
  trades performance for something else must say so and include
  before/after numbers.

Rule 1 is the main way to meet this rule: if the disabled path is
untouched, it cannot regress.

## Open questions / to decide

- Exceptions to rule 1: refactors, or bug fixes that must change the
  existing path. These presumably need their own PR, stated
  explicitly.
- How a contributor checks for performance regressions: which
  benchmarks, and whether CI covers it.
- Where these rules live in the end: `CONTRIBUTING.md`, with
  `AGENTS.md` pointing to it.
