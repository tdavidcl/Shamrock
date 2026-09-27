# Making a "Satisfying astro simulations" YouTube Short with Shamrock

This is a playbook for producing one YouTube Short end to end: simulate something with
Shamrock, render it cinematically, encode a 1080×1920 30 s video, and publish it as a
private claude.ai Artifact with download buttons.

The playbook fixes the **process**, not the **scene**. Each run must invent its own
physical setup (see step 1).

---

## Deliverables

1. `short_final.mp4`: 1080×1920, 30 fps, exactly 30.0 s, H.264 `yuv420p`, `+faststart`,
   no text overlays, no audio.
2. `thumbnail.png`: a full-resolution frame from the most striking moment.
3. A private Artifact page that plays the video and has buttons to download the MP4 and
   the thumbnail.
4. Previews sent to the user every 5 min of simulation wallclock (see step 5).

## Hard constraints from the channel owner

- Use the **SPH** solver. If the scene has point masses (stars, planets, black holes),
  they must be **sink particles** (`model.add_sink`). Point masses are optional.
- If the camera moves, it is **one continuous smooth movement** over the whole video: a
  single eased sweep, never cuts or several separate moves.
- The video must look fancy, stylish and satisfying, and be engaging from the first
  second (hook), because click-through and retention matter.
- Target **between 20 min and 1 h of simulation wallclock**. If the run is heading past
  1 h, say so in the previews.
- If the result is not satisfying enough, change the setup and run it again.

---

## 1. Pick a scene that is new

Choose the physical scenario yourself, and **make it different from earlier shorts**:

- List the user's artifacts (`Artifact` tool, `action: "list"`) and read the titles and
  descriptions of earlier shorts. Pick a phenomenon, geometry and colour mood that has not
  been done yet. If earlier descriptions mention a scenario, treat it as used up.
- Good shorts have **one clear story arc** in 30 s: a calm or recognisable start, a
  dramatic event, then a visually rich aftermath (spirals, filaments, shells, streams,
  rings, turbulence, mixing).
- Structure has to appear within the first 1–3 s of the video.
- Favour setups that stay affordable on the hardware you have (step 3). Scenes with
  extreme dynamic range in timescale are expensive.
- Browse `examples/sph/` for the setup tools Shamrock offers: lattice and disc
  generators, and the offset, filter, custom-warp (rigid rotation) and split modifiers,
  combined with `make_combiner_add`. You can also use custom fields through
  `set_field_value_lambda_*`, `add_kill_sphere`, several EOS options, and external
  forces. Combine them creatively.

Write the choice down (1–2 sentences) before building. You will reuse it in the Artifact
description, which is how the next run knows what was already done.

## 2. Build Shamrock (cloud container, CPU only)

Follow `AGENTS.md` and `CLAUDE.md`:

- If `build/shamenv_do` already exists, **don't** re-run `./env/new-env`.
- Build only the `shamrock` executable. It pulls in everything the Python runscript
  needs:
  ```bash
  cd build
  ./shamenv_do shamconfigure        # first time also builds AdaptiveCpp (minutes)
  ./shamenv_do shammake shamrock && echo BUILDDONE
  ```
  On 4 cores a cold build takes about 60–80 min. Start it in the background immediately,
  and write the simulation and the renderer while it compiles.
- Install the render tooling in parallel: `apt-get install -y ffmpeg`,
  `pip install numpy scipy numba matplotlib pillow`.
- Run scripts with
  `./shamenv_do ./shamrock --sycl-cfg 0:0 --loglevel 1 --rscript sim.py`.
  If `--smi` shows more than one device, ask the user which one to use (once).

## 3. Write the simulation script (`sim.py`)

- Keep all physical and numerical parameters in a JSON file read by the script
  (`SIM_PARAMS=params.json`), so relaunching with changes needs no code edits.
- Use code units that suit the scene (`shamrock.UnitSystem`). Get `G` from
  `shamrock.Constants(codeu).G()`.
- Solver config essentials: EOS, artificial viscosity, units, particle mass, CFL
  (`set_cfl_cour(0.3)`, `set_cfl_force(0.25)`),
  `set_smoothing_length_density_based_neigh_lim(500)` (avoids giant-h slowdowns), and a
  kill sphere well inside a generous simulation box.
- **Per-particle identity** (colouring by origin or material): enable
  `shamrock.enable_experimental_features()`, then `cfg.set_particle_tracking(True)`.
  `ctx.collect_data()` then returns `part_id`. Ids follow **generation order**: with
  `make_combiner_add(g1, g2)`, the particles of `g1` come first. Tag by id range and save
  a `tag.npy`. Never tag by proximity, because overlapping regions produce hard
  seams.
- Many pybind setters are **keyword-only** (for example
  `set_eos_locally_isothermalFA2014(h_over_r=...)`). If you get an "incompatible
  function arguments" error, check the binding in `src/shammodels/sph/src/pySPHModel.cpp`.
- Initialise with `change_htolerances(coarse=1.3, fine=1.1)`, one `model.timestep()`,
  then restore `(1.1, 1.1)`.
- **Snapshots for offline rendering**: every `dt_out` of sim time, call
  `model.evolve_until(i*dt_out)`, then `ctx.collect_data()`. Store float32 arrays
  **indexed by particle id**, full length N: `xyz` (NaN for removed particles), `v`,
  `h` (0 for removed particles). Also store the sinks (pos, vel, mass) and `t`. Write to
  a temp file and `os.replace` it, so readers never see partial files. Print a
  `[SNAP] i/n t=... wall=...` line after each dump for monitoring.
  Aim for about 150–350 snapshots. The renderer interpolates between them.

### Budget the run before committing to it

- Shamrock SPH throughput on the 4-core CPU container (OpenMP backend) is about
  **2–2.5·10⁵ particle-steps/s**.
- The timestep is **global**. The smallest dynamical time in the whole domain sets the
  cost: gas close to a sink, dense cores, fast shocks. The log prints
  `dt = ...`, `cfl multiplier`, and the per-step `tstep`/`rate`.
- Step count is roughly the sim duration divided by the typical dt.
  Wallclock is roughly `steps × N / rate`.
- The corrector rejects some steps: "corrector tolerance are broken ... re rerunned"
  warnings cost extra steps. Encounters, collisions and shocks make dt collapse
  mid-run. Expect a 2–4× slowdown during the dramatic phase.
- **Always run a short calibration first** (a few hundred steps). If the projection
  exceeds the budget by a lot, fix the setup, not the patience:
  - raise sink accretion radii or inner cut-offs;
  - shorten the covered sim time and start closer to the action;
  - lower N (60k–100k particles already render smoothly);
  - soften extreme density or velocity contrasts.

## 4. Renderer (`render.py`), offline and independent of the sim

Render from snapshots, not inside Shamrock. You can then tune the camera, colours and
timing freely without re-simulating.

- **Splatting**: project particles orthographically onto the camera plane. Deposit the
  column-integrated SPH kernel (the M4 kernel integrated along the line of sight,
  normalised, as a 1D lookup table in q∈[0,2]) with weight `1/h²`.
  - Clamp `h` to at least ~0.6 px so tiny particles don't alias.
  - Use numba with `parallel=True` and **one image buffer per thread**, summed at the
    end (no atomics).
  - Scale by px²/au² so values don't depend on zoom.
- **Temporal interpolation**: cubic **Hermite** between the two bracketing snapshots,
  using positions and velocities. The result is smooth motion at 30 fps from sparse
  snapshots, with no ghosting. Only draw particles alive in both snapshots. Interpolate
  the sinks the same way.
- **Colour** (choose a palette that fits the scene, not the previous short):
  - Tone-map `log10(column density)` between fixed `lo`/`hi` values. Keep them constant
    across frames, or it will flicker.
  - A **local-contrast** term in log space, `x + k·(x − blur(x))`, makes arms, edges and
    filaments pop.
  - If you colour by tag or material, compute a **smoothed fraction**
    `blur(Σ_tag)/blur(Σ_total)` and blend palettes by it. Do not tone-map each channel
    separately: isolated particles of one kind inside another turn into speckles.
- **Glamour pass**:
  - two-scale bloom, computed at quarter resolution;
  - filmic tone map `1 − exp(−exposure·rgb)`;
  - a soft vignette and a very dark tinted background;
  - sink particles drawn as a bright core with a halo and a faint horizontal streak;
  - a sparse **starfield placed on the celestial sphere**, projected with the camera
    basis so it moves consistently with the camera, and hidden behind bright gas.
- **Direction plan** (`plan.json`):
  - `time_keys`: a video-time → sim-time mapping through a PCHIP interpolant (smooth).
    Speed up quiet phases, slow down the climax, and start just before something
    visibly happens.
  - Camera: azimuth, elevation, roll, half-width (log-interpolated zoom) and centre.
    All of them are driven by **one** eased parameter, a mix of smootherstep and linear
    (`cam_ease` ≈ 0.7), so the start isn't frozen.
  - For a 9:16 frame, orient the camera so the scene's long axis is **vertical**.
  - Check framing with a small planner script that projects the sink tracks (or gas
    percentiles) at sampled frames, before rendering anything big.
- **Output**: pipe raw RGB frames straight into ffmpeg (`libx264 -preset slow -crf 14–16
  -pix_fmt yuv420p -profile:v high -movflags +faststart`). Support `--f0/--f1` frame
  ranges, then render 3 chunks in parallel and join them with the concat demuxer
  (`-c copy`).
- Also support `--still N` (one PNG), `--stride`, a low-res `--W/--H`, and
  `--upto_available` (only frames whose sim time already exists) for previews.
- Review tools: a **contact sheet**, one low-res frame every ~2.5 s tiled in a row. It
  is the fastest way to judge the story arc and framing. Look at it before every full
  render.

## 5. Run the simulation with live previews

- Launch the sim in the background with `nohup`. Record the start time. Watch it with a
  loop on the `[SNAP]` / `Traceback` / `what()` lines.
- **Never** run `pkill -f "<pattern>"` where the pattern also appears in your own shell
  command line, because it kills your own shell. Match on something only the sim's
  command line contains, or kill by PID.
- **Every 5 min of simulation wallclock**:
  - render a preview (`--W 360 --H 640 --stride 2 --upto_available`, `nice`-d so it
    doesn't starve the sim);
  - send it with `SendUserFile` (`status: "proactive"`), with a one-line caption: sim
    time reached, what is happening, and any runtime or framing concern.
- Use the previews to refine the look and camera plan while the sim runs. Decide the
  end time from what you see and from the measured speed. Stop the sim once the
  aftermath looks complete.
- If the previews show the setup isn't satisfying (boring, broken, too slow), stop it,
  change the setup, and run again. Tell the user why.

## 6. Final render and checks

- Final plan: frame the whole arc (contact sheet), glow on the key objects, and a hook in
  the first second.
- Render at 1080×1920, 900 frames, in 3 parallel chunks, then concatenate.
- Verify with `ffprobe`: 1080×1920, 30/1 fps, 900 frames, duration 30.000000.
- Build a contact sheet of the final file (`fps=0.5,scale=216:384,tile=15x1`) and
  inspect it.
- Extract `thumbnail.png` at the most striking moment
  (`ffmpeg -ss <t> -i short_final.mp4 -frames:v 1 thumbnail.png`).
- Send the MP4 and thumbnail to the user with `SendUserFile`.

## 7. Publish the Artifact with download buttons

1. `Artifact` with `action: "quickstart"`, `intent: "other"`: a plain page, no type.
2. Load the `artifact-capabilities` skill. The page needs the `downloads` capability.
   Plain `<a download>` links are blocked in the viewer.
3. Write one HTML file in the scratchpad:
   - `<title>`: a short, distinctive name for this short (2–4 words).
   - A 9:16 `<video src="short_final.mp4" poster="thumbnail.png" controls loop muted
     playsinline autoplay>`.
   - A one-paragraph description of what happens, a small spec list (format, length,
     solver, particle count, sinks, sim time span), and two buttons.
   - Button handler:
     ```js
     const downloads = await window.claude?.use?.("downloads"); // null → disable buttons
     const blob = await (await fetch("short_final.mp4")).blob();
     await downloads.save({ filename: "<name>.mp4", data: blob });
     ```
     Handle `declined` and `rate_limited`, and disable the buttons when `downloads` is
     `null`.
   - Style it after the video's own palette. A single dark "screening room" theme is
     fine, as long as `body` gets an explicit background.
4. Publish with the following:
   - `icon: "video"`
   - `capabilities: {"downloads": true}`
   - `files: {"short_final.mp4": "<abs path>", "thumbnail.png": "<abs path>"}`
   - a one-sentence `description` that **names the scenario**. The next run reads it in
     step 1 to avoid repeating the scenario.
5. Give the user the artifact URL and mention that it is private until they share it.

## 8. Wrap-up message to the user

Keep it short:

- what the scene is and its story arc;
- solver settings (particle count, sinks, sim time covered);
- actual sim wallclock against the 20 min – 1 h target;
- any restarts or tweaks and why;
- where the files are, and the Artifact link.

Don't commit render outputs or snapshot data to the repository.
