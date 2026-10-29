#!/usr/bin/env python3
"""
Adreno 830 without wave128: supports_double_threadsize = False, as upstream.

a8xx_gen8.patch turns supports_double_threadsize on in a8xx_base, i.e. for
every A8xx. Upstream Mesa turned it off for A8xx in eb43e95 (Rob Clark,
"Gone is thread128. Instead the hw can co-dispatch thread64"). With it off,
compute always dispatches THREAD128 (a co-dispatched pair of wave64s), the
compiler stops doubling max workgroup invocations, and the driver no longer
exposes subgroup size 128.

This script turns it off for the A830 only (gen8_3_0, the S25 Ultra's GPU)
by appending GPUProps(supports_double_threadsize = False) to the A830 entry's
template list. GPUProps.apply_props runs in list order, so the last template
wins and a8xx_base stays as it is for every other A8xx.

Runs from the Mesa source root, after a8xx_gen8.patch. Idempotent; fails
loudly if the A830 entry is not where it expects it.
"""
import sys

DEVICES_PY = "src/freedreno/common/freedreno_devices.py"
MARKER = 'GPUId(chip_id=0x44050001, name="Adreno (TM) 830")'
OVERRIDE = "GPUProps(supports_double_threadsize = False)"


def die(msg):
    print(f"a830_wave64.py: FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


with open(DEVICES_PY) as f:
    src = f.read()

if src.count(MARKER) != 1:
    die(f"expected one A830 KGSL entry {MARKER}, found {src.count(MARKER)}")

start = src.index(MARKER)
info = src.find("A6xxGPUInfo(", start)
next_entry = src.find("add_gpus(", start)
if info == -1 or (next_entry != -1 and info > next_entry):
    die("no A6xxGPUInfo( for the A830 entry")

# The template list is the first [...] inside A6xxGPUInfo(.
open_br = src.index("[", info)
depth = 0
close_br = -1
for i in range(open_br, len(src)):
    if src[i] == "[":
        depth += 1
    elif src[i] == "]":
        depth -= 1
        if depth == 0:
            close_br = i
            break
if close_br == -1:
    die("unbalanced A830 template list")

templates = src[open_br:close_br + 1]
if "a8xx_base" not in templates:
    die(f"A830 template list does not include a8xx_base: {templates}")
if "supports_double_threadsize" in templates:
    print("a830_wave64.py: A830 already sets supports_double_threadsize, nothing to do")
    sys.exit(0)

# Insert after the last template, keeping any trailing comma valid.
end = close_br
while src[end - 1].isspace():
    end -= 1
sep = " " if src[end - 1] == "," else ", "
src = src[:end] + sep + OVERRIDE + src[end:]

try:
    compile(src, DEVICES_PY, "exec")
except SyntaxError as e:
    die(f"syntax error after the edit at line {e.lineno}: {e.msg}")

with open(DEVICES_PY, "w") as f:
    f.write(src)

new_close = src.index("]", end + len(sep) + len(OVERRIDE) - 1)
print(f"a830_wave64.py: A830 templates now {' '.join(src[open_br:new_close + 1].split())}")
