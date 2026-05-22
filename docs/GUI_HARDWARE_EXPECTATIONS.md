# GUI Hardware Expectations

Metal PathTracer is usable across Apple Silicon systems, but the interactive GUI
has a much larger live working set than a focused headless render. This document
sets practical expectations for large asset-pack scenes such as Bistro, San
Miguel, Sponza, and repeated high-density mesh scenes such as multi-instance
Lucy.

These tiers are conservative guidance, not strict guarantees. Actual behavior
depends on scene selection, texture resolution, render resolution, direct
lighting mode, ReSTIR state, denoising, other open applications, macOS memory
pressure, and whether the device can use the preferred Metal ray tracing path.

## Summary

- 16 GB unified memory is not enough to guarantee that every asset-pack scene
  can load in the GUI. An M1 Pro MacBook Pro with 16 GB has been observed failing
  to load a Lucy scene with three model instances.
- More than 16 GB unified memory should be treated as the practical minimum for
  full-scene GUI bring-up.
- 24 GB unified memory is a transitional tier: useful for some large-scene
  inspection, but still constrained for repeated heavy scene cycling, denoising,
  and ReSTIR modes.
- 32 GB unified memory should be treated as the first
  broadly useful tier for large-scene GUI bring-up. Keep in mind, though, that Apple Silicon MacBook Air systems lack active cooling, so
  sustained GUI rendering, denoising, shader compilation, and asset processing
  can throttle even when memory capacity is sufficient.
- 36 GB or 48 GB unified memory is the more realistic practical baseline for
  loading and inspecting large scenes on current Pro/Max-class systems, but the
  GUI can still be slow under pressure.
- 64 GB unified memory is recommended for comfortable large-scene GUI and
  development work, especially on Max-class systems.
- Headless rendering has different expectations because it avoids interactive UI
  overhead and is easier to run with controlled resolution, sample count, and
  output settings.

## Apple Silicon Memory Configurations

Apple Silicon memory tiers are tied to chip and product class; they are not a
simple power-of-two ladder. Current and recent Apple Silicon Macs include these
unified-memory capacities:

- Mainstream/base-chip systems: 8 GB on earlier systems; 16 GB, 24 GB, and
  32 GB on current MacBook Air, MacBook Pro, iMac, and some Mac mini
  configurations.
- Pro-class systems: 16 GB and 32 GB on earlier M1/M2 Pro systems; 18 GB and
  36 GB on M3 Pro systems; 24 GB, 48 GB, and 64 GB on current higher-end Pro
  configurations where offered.
- Max-class systems: 32 GB, 36 GB, 48 GB, 64 GB, 96 GB, and 128 GB depending on
  generation, GPU bin, and form factor.
- Ultra-class systems: 64 GB, 96 GB, 128 GB, 192 GB, and higher capacities on
  some Mac Studio generations, subject to the exact chip and current CTO
  availability.

For this renderer, capacity is only one axis. A 32 GB base-chip Mac can have
more memory than a 24 GB Pro system, but the Pro/Max system may still have more
GPU throughput, memory bandwidth, and sustained thermal headroom. Fanless
MacBook Air systems are useful for small-scene look development and occasional
large-scene inspection, but they should not be treated as equivalent to actively
cooled Pro/Max systems for long GUI sessions. Use the tier table below for GUI
expectations rather than treating any single capacity as a complete performance
class.

## Practical Tiers

| System class | Unified memory | GUI expectation |
| --- | ---: | --- |
| Base M1/M2/M3-class systems | 8 GB | Build and small-scene experimentation only. Large asset-pack scenes should be expected to fail, page heavily, or become unusable in the GUI. |
| M1/M2 or M1 Pro-class systems | 16 GB | Suitable for small and moderate scenes. Not sufficient as a minimum target for loading every asset-pack scene in the GUI. Large repeated-mesh scenes and Bistro/San Miguel/Sponza-class scenes are not guaranteed. |
| Pro/Max systems | 24 GB | Transitional tier. Some large scenes may load with `PT_HWRT_BUILD_POLICY=compact_memory`, but GUI responsiveness and ReSTIR modes are still constrained. |
| Base/Pro/Max systems | 32 GB | First broadly useful large-scene GUI tier. 32 GB exists on Apple Silicon Macs, but chip class and cooling still matter: fanless MacBook Air systems can throttle under sustained renderer load, and base-chip systems have less GPU throughput and memory bandwidth than Pro/Max systems. |
| Pro/Max systems | 36 GB to 48 GB | Practical baseline for large-scene GUI bring-up on current Pro/Max-class hardware. Large scenes can still consume most available memory and run at low interactive frame rates. |
| Max-class systems | 64 GB | Recommended for comfortable large-scene GUI and development work. Current high-end Max systems with 64 GB have enough headroom for the heaviest included models under typical development use. |
| Ultra/high-memory systems | 96 GB or more | Extra headroom for unusually large render targets, multiple heavy applications, parallel tooling, or future scene expansion; not required as the baseline development target. |

## Why the GUI Needs More Memory

Apple Silicon uses unified memory: CPU memory, GPU resources, Metal acceleration
structures, textures, render targets, denoiser buffers, and ReSTIR reservoirs all
come from the same physical memory pool. The GUI also keeps interactive renderer
state resident while macOS, the window server, development tools, and other apps
continue using memory.

`PT_HWRT_BUILD_POLICY=compact_memory` is recommended for large static scenes
because it reduces Metal acceleration-structure pressure. It does not reduce
texture memory, imported scene buffers, render targets, denoiser state, or
ReSTIR reservoirs.

Interactive OIDN denoising also needs GPU-to-CPU readback and CPU-to-GPU
writeback buffers. Under high Metal residency, the GUI may skip or defer OIDN
attempts rather than repeatedly allocating transfer work. Use headless rendering
for final denoised output when a large interactive session is already near the
device working-set limit.

Long-running GUI sessions are guarded against excessive pressure after many
scene switches, especially when moving between large scenes while ReSTIR,
denoising, and interactive transport resources are active. Current builds force
each scene switch as an explicit release point for the previous scene, reuse the
renderer command queue and bounded staging buffers for OIDN transfers, and defer
OIDN attempts while residency is materially beyond
`MTLDevice.recommendedMaxWorkingSetSize`. A small hysteresis band avoids
disabling OIDN for tiny transient overshoots and allows denoising to resume
automatically after residency drops. These safeguards avoid the repeated OIDN
readback/writeback failure loop seen under pressure, but they do not make OIDN
free: on lower-headroom systems, denoising may still be temporarily skipped
until residency drops. A 36 GB Max-class system can expose this behavior during
extended large-scene cycling, while a 64 GB system may have enough working-set
headroom to avoid it under the same workflow.

Chip generation, GPU class, and cooling also matter. Memory capacity mainly
determines whether a scene can be loaded and kept resident. GPU throughput,
memory bandwidth, the available Metal ray tracing path, and sustained thermal
headroom determine whether the GUI feels responsive after the scene loads.

## Headless Versus GUI

The headless renderer should not be judged by the same requirements as the GUI.
Headless runs can be configured with explicit resolution, sample count, render
profile, output path, and validation settings. They do not carry the same
interactive UI overhead and are usually the better option for final output on
memory-constrained systems.

For systems at or below 16 GB unified memory, prefer headless rendering for large
scenes and use the GUI for smaller look-development scenes.

## Recommended Large-Scene GUI Launch

```bash
PT_HWRT_BUILD_POLICY=compact_memory \
./build-metal/PathTracer.app/Contents/MacOS/PathTracer --scene=bistro_full
```

Enable the explicit interactive ReSTIR DI path only when profiling that path
specifically:

```bash
PT_HWRT_BUILD_POLICY=compact_memory \
PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI=1 \
./build-metal/PathTracer.app/Contents/MacOS/PathTracer --scene=bistro_full
```

## Operator Guidance

- Close memory-heavy applications before loading large scenes in the GUI.
- Start with smaller scenes before loading Bistro, San Miguel, Sponza, or
  repeated high-density mesh scenes.
- Use `PT_HWRT_BUILD_POLICY=compact_memory` for large static scenes.
- Avoid explicit interactive ReSTIR DI on constrained systems unless that mode is
  the thing being tested.
- Use headless rendering for final output when the GUI is slow but the scene can
  still be rendered offline.
