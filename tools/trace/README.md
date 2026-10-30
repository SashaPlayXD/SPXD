# Trace tools

Two scripts for reading traces recorded on the phone, without root.

## GPU timeline: `gpu_trace_summary.py`

Record with two environment variables in the WinNative shortcut:

```
MESA_GPU_TRACES=print_csv
MESA_GPU_TRACEFILE=/storage/emulated/0/Download/spxd_gpu_%p.csv.gz
```

`%p` becomes the process id, and the `.gz` ending makes the driver compress
the file itself. Ten seconds of gameplay is enough; a whole benchmark run
gives a file too big to send from a phone. Then:

```
python3 tools/trace/gpu_trace_summary.py spxd_gpu_1234.csv.gz
python3 tools/trace/gpu_trace_summary.py r4.csv.gz --compare r10.csv.gz
python3 tools/trace/gpu_trace_summary.py spxd_gpu_1234.csv.gz --anatomy
```

It prints GPU time per frame, how busy the GPU is between frames, and the
render passes, compute shaders, blits and LRZ-off reasons that cost the
most. `--compare` puts a second trace beside the first (same save, other
build); `--anatomy` lists one median frame in GPU order.

Recording costs a noticeable share of FPS, so take benchmark numbers with
the two variables removed.

Under WinNative the trace's frame column stays 0, because frames are
presented outside Turnip. The script then counts frames by vkd3d-proton's
present blit (the most common single-draw, single-attachment command buffer,
or the size given with `--present 1300x600`). That blit does the same work
every frame, so its duration, the "blit us" column of the timeline, tracks
the GPU clock: about 87 us at full clock on the S25 Ultra, 460 us in menus,
170-190 us when the phone throttles.

To tell builds apart from the trace alone: a compute shader's `cs_sha1`
covers its wave size too, so shaders that r4 ran as wave128 carry different
hashes under r10 wave64.

## Android system trace: `systrace_summary.py`

Record with Developer options -> System Tracing (CPU Scheduling and CPU
Frequency on), with the FPS limiters off. Then:

```
pip install perfetto protobuf
python3 tools/trace/systrace_summary.py trace.perfetto-trace
```

It prints CPU time per process, and for each game thread how much of the
window it ran, waited for a CPU, or slept, who woke it, and the clock of
the cores it ran on; then per-core load and clocks, the GPU clock if it was
recorded, and the display frame rate. Only the Perfetto protos from the
`perfetto` package are used, not its trace_processor binary.
