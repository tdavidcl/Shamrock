# `shamunits` redesign — session export

Design work for rebuilding `src/shamunits/` around C++20 values instead of
X-macros. **Nothing in `src/shamunits/` has been changed yet** — this directory
is the design record produced before implementation.

| File | What it is |
| --- | --- |
| `01-conversation.md` | How the design was reached: the request, what the survey found, every design fork and the decision taken at it, and the two compiler measurements. Read this for *why*. |
| `02-plan.md` | The implementation plan: final design, file-by-file changes, ordered steps, verification. Read this for *what to do*. |
| `03-compiler-probes.md` | What the two probe programs test, how to run them, and the results measured so far. |
| `nttp_probe.cpp` | Standalone C++20 probe: does the NTTP form compile, and in which spelling. Answered. |
| `codegen_probe.cpp` | Standalone C++20 probe: what the planned `get` actually emits. Measured on gcc 13 and clang 20. |

Both `.cpp` files are self-contained and godbolt-ready — `-std=c++20 -O2`, no
project headers.

## Status

- Design: settled, with the user's decisions recorded in `01-conversation.md`.
- NTTP feasibility: **measured** — `template<const Unit &u>` works on clang 15;
  by-value `template<Unit u>` needs clang ≥ 18.
- Codegen: **measured** on gcc 13 / clang 20 — the NTTP form emits one
  instruction for a simple unit and keeps one division out of seven in the
  realistic case. One comparison against the current `addget` remains.
- Implementation: not started.

## Plan revision history

Three snapshots of the plan surfaced while assembling this export. `02-plan.md`
is the newest. If another copy turns up, place it by these markers:

| Marker in the file | Revision |
| --- | --- |
| `get<units::astronomical_unit, 2>()`, `runtime_get`, `ipow_pos` | oldest — `power` still a separate template parameter |
| "Risk: this requires a by-value `Unit` NTTP", "Resolve this in step 0" | middle — power folded into the unit, NTTP question still open |
| "Measured on clang 15 and 18", `template<const Unit &u>` | **current** — this file |

Only one thing was lost across those revisions and has been restored here: the
rationale for why a reference NTTP is legal (external linkage / static storage
duration). The dropped `ipow_pos`, `runtime_get`-via-`std::pow`, and the
`get<p>(u) == get<1>(u)^p` identity are all obsolete — they existed only to
support the separate `power` parameter, which no longer exists.
