#!/usr/bin/env python3
"""
Check one prop of one GPU in a generated freedreno_devices.h.

usage: check_dev_info.py <freedreno_devices.h> <chip_id> <prop> <True|False>

freedreno_devices.py writes one `static const struct fd_dev_info __infoN =
{...}` line per distinct GPU and a table of `{ {0, <chip_id>}, "<name>",
&__infoN }` rows. This finds the row for <chip_id>, then <prop> in its
__infoN, so a build can prove what the driver it ships was compiled with.
"""
import re
import sys

if len(sys.argv) != 5:
    sys.exit(__doc__.strip().splitlines()[2])

path, chip_id, prop, expected = sys.argv[1:]
src = open(path).read()

row = re.search(r"\{\s*\{\s*0\s*,\s*" + re.escape(chip_id) + r"\s*\}\s*,\s*\"([^\"]*)\"\s*,\s*&(__info\d+)\s*\}", src)
if not row:
    sys.exit(f"check_dev_info.py: no table row for chip_id {chip_id} in {path}")
name, info = row.groups()

body = re.search(r"static const struct fd_dev_info " + info + r" = (\{.*?\});\s*$", src, re.M)
if not body:
    sys.exit(f"check_dev_info.py: {info} ({name}) not found in {path}")

val = re.search(r"\." + re.escape(prop) + r"=([A-Za-z0-9_]+)", body.group(1))
if not val:
    sys.exit(f"check_dev_info.py: {prop} not set for {name} ({chip_id}, {info})")
if val.group(1) != expected:
    sys.exit(f"check_dev_info.py: {name} ({chip_id}): {prop}={val.group(1)}, expected {expected}")

print(f"check_dev_info.py: {name} ({chip_id}, {info}): {prop}={val.group(1)}")
