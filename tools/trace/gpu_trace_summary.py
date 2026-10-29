#!/usr/bin/env python3
"""
Summarize a Turnip GPU timeline recorded with MESA_GPU_TRACES=print_csv.

usage: gpu_trace_summary.py TRACE [--compare TRACE2] [--frames A-B] [--ctx N]
                            [--top N] [--anatomy]

TRACE is the file MESA_GPU_TRACEFILE named: .csv, or .csv.gz when the name
ends in .gz. A gzip stream cut short (the game was killed before exit) is
read up to the cut.

Recording on the phone, no root:
  MESA_GPU_TRACES=print_csv
  MESA_GPU_TRACEFILE=/storage/emulated/0/Download/spxd_gpu_%p.csv.gz
%p becomes the process id, so every process that loads the driver writes its
own file; the game's is the big one.

Each line is "frame,batch,time_ns,event," followed by the tracepoint's
"key=value, key=value" payload and "ctx=N" last (src/util/perf/u_trace.c,
print_csv_event). start_X/end_X pairs give the GPU time of each command
buffer, render pass, compute dispatch, blit, copy and clear.

By default the analysis takes the longest run of "heavy" frames (at least
half the render passes of a typical gameplay frame), which skips menus and
loading screens. --frames picks a range by hand; the per-second timeline at
the top shows what is where.

--compare prints the same numbers for a second trace beside the first, for
example r4 against r10 wave64 recorded at the same save.

Without MESA_GPU_TRACES' "indirects" flag, lrzStatus and the indirect
dispatch sizes are placeholders and are not used.
"""
import argparse
import collections
import gzip
import re
import statistics
import sys
import zlib

KEY = re.compile(r'(?:^|, )([A-Za-z_][A-Za-z0-9_]*)=')

# Ranges nested inside a render pass: counted inside it, never on their own.
IN_RENDER_PASS = {
    'draw', 'binning_ib', 'concurrent_binning_ib', 'draw_ib_sysmem',
    'draw_ib_gmem', 'concurrent_binning_barrier', 'generic_clear',
    'gmem_clear', 'sysmem_clear', 'sysmem_clear_all', 'gmem_load',
    'gmem_store', 'sysmem_resolve', 'custom_resolve',
    'cmd_buffer_annotation_rp',
}
TRANSFER = {
    'blit_image', 'clear_color_image', 'clear_depth_stencil_image',
    'copy_buffer_to_image', 'copy_image_to_buffer', 'copy_image',
    'resolve_image', 'fill_buffer', 'copy_buffer', 'update_buffer',
    'slow_clear_lrz', 'disable_lrz',
}


def parse_kv(s):
    """'a=1, b=x y, ctx=0' -> {'a': '1', 'b': 'x y', 'ctx': '0'}."""
    out = {}
    ms = list(KEY.finditer(s))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(s)
        out[m.group(1)] = s[m.end():end].strip()
    return out


def open_lines(path):
    """Yield text lines; a truncated .gz ends quietly with a warning."""
    if path.endswith('.gz'):
        f = gzip.open(path, 'rt', encoding='utf-8', errors='replace')
    else:
        f = open(path, encoding='utf-8', errors='replace')
    with f:
        try:
            for line in f:
                yield line
        except (EOFError, zlib.error, OSError) as e:
            print(f'note: {path} ends early ({e}); using what was written',
                  file=sys.stderr)


class Frame:
    __slots__ = ('nr', 'first_ts', 'cb_ns', 'cb_spans', 'n_rp', 'n_disp',
                 'cat_ns', 'rp_ns', 'rp_n', 'rp_draws', 'rp_gmem',
                 'cs_ns', 'cs_n', 'cs_groups', 'tr_ns', 'tr_n', 'ann_ns',
                 'in_rp_ns', 'lrz_off_ns', 'lrzw_off_ns', 'mode_reason_n')

    def __init__(self, nr):
        self.nr = nr
        self.first_ts = None
        self.cb_ns = 0
        self.cb_spans = []
        self.n_rp = 0
        self.n_disp = 0
        self.cat_ns = collections.Counter()
        self.rp_ns = collections.Counter()
        self.rp_n = collections.Counter()
        self.rp_draws = collections.Counter()
        self.rp_gmem = collections.Counter()
        self.cs_ns = collections.Counter()
        self.cs_n = collections.Counter()
        self.cs_groups = collections.Counter()
        self.tr_ns = collections.Counter()
        self.tr_n = collections.Counter()
        self.ann_ns = collections.Counter()
        self.in_rp_ns = collections.Counter()
        self.lrz_off_ns = collections.Counter()
        self.lrzw_off_ns = collections.Counter()
        self.mode_reason_n = collections.Counter()


def rp_signature(a):
    depth = 'depth' if a.get('hasDepth') == 'true' else 'no depth'
    return (f"{a.get('width', '?')}x{a.get('height', '?')} "
            f"att={a.get('attachment_count', '?')} {depth} "
            f"samples={a.get('maxSamples', '?')}")


def cs_signature(name, a):
    sha = a.get('cs_sha1', '?')[:12]
    if name == 'compute_indirect':
        return f'{sha} indirect'
    return (f"{sha} wg={a.get('local_size_x', '?')}x{a.get('local_size_y', '?')}"
            f"x{a.get('local_size_z', '?')}")


def groups_of(a):
    try:
        return (int(a['num_groups_x']) * int(a['num_groups_y']) *
                int(a['num_groups_z']))
    except (KeyError, ValueError):
        return 0


def load(path, want_ctx=None):
    """Parse one trace. Returns (frames by nr, ctx event counts, info)."""
    per_ctx_lines = collections.Counter()
    events = []  # (ctx, frame, ts, name, rest) of the chosen context
    bad = 0
    app = None
    for line in open_lines(path):
        if not line or line[0] not in '0123456789':
            continue
        parts = line.rstrip('\n').split(',', 4)
        if len(parts) < 5:
            bad += 1
            continue
        rest = parts[4]
        i = rest.rfind('ctx=')
        if i < 0:
            bad += 1
            continue
        try:
            ctx = int(rest[i + 4:])
            frame = int(parts[0])
            ts = int(parts[2])
        except ValueError:
            bad += 1
            continue
        per_ctx_lines[ctx] += 1
        events.append((ctx, frame, ts, parts[3], rest[:i].rstrip(', ')))

    if not events:
        sys.exit(f'{path}: no trace events found')
    ctx = want_ctx if want_ctx is not None else per_ctx_lines.most_common(1)[0][0]

    frames = {}
    stack = []  # open ranges of this context: [name, ts, args, frame]
    unmatched = 0

    def fr(nr):
        f = frames.get(nr)
        if f is None:
            f = frames[nr] = Frame(nr)
        return f

    for c, frame, ts, name, payload in events:
        if c != ctx:
            continue
        if name.startswith('start_'):
            base = name[6:]
            args = parse_kv(payload) if payload else {}
            if base == 'cmd_buffer' and app is None:
                app = args.get('appName')
            stack.append([base, ts, args, frame])
            f = fr(frame)
            if f.first_ts is None or ts < f.first_ts:
                f.first_ts = ts
            continue
        if not name.startswith('end_'):
            continue  # singular tracepoints (perf warnings)
        base = name[4:]
        j = len(stack) - 1
        while j >= 0 and stack[j][0] != base:
            j -= 1
        if j < 0:
            unmatched += 1
            continue
        unmatched += len(stack) - 1 - j
        sbase, sts, sargs, sframe = stack[j]
        del stack[j:]
        dur = max(0, ts - sts)
        f = fr(sframe)
        if base == 'cmd_buffer':
            f.cb_ns += dur
            f.cb_spans.append((sts, ts))
            continue
        if base == 'render_pass':
            eargs = parse_kv(payload) if payload else {}
            sig = rp_signature(sargs)
            gmem = eargs.get('tiledRender') == 'true'
            f.n_rp += 1
            f.cat_ns['render pass, gmem' if gmem else 'render pass, sysmem'] += dur
            f.rp_ns[sig] += dur
            f.rp_n[sig] += 1
            f.rp_gmem[sig] += gmem
            try:
                f.rp_draws[sig] += int(eargs.get('drawCount', 0))
            except ValueError:
                pass
            reason = eargs.get('lrzDisableReason')
            if eargs.get('lrz') == 'false' and reason and reason != '(null)':
                f.lrz_off_ns[reason] += dur
            reason = eargs.get('lrzWriteDisableReason')
            if reason and reason != '(null)':
                f.lrzw_off_ns[reason] += dur
            reason = eargs.get('forceRenderModeReason')
            if reason and reason != '(null)':
                f.mode_reason_n[reason] += 1
            continue
        if base in ('compute', 'compute_indirect'):
            sig = cs_signature(base, sargs)
            f.n_disp += 1
            f.cat_ns[base] += dur
            f.cs_ns[sig] += dur
            f.cs_n[sig] += 1
            f.cs_groups[sig] += groups_of(sargs)
            continue
        if base in IN_RENDER_PASS:
            key = base
            if base == 'cmd_buffer_annotation_rp':
                key = 'label: ' + parse_kv(payload).get('str', '?')
            f.in_rp_ns[key] += dur
            continue
        if base in TRANSFER:
            fmt = sargs.get('format') or sargs.get('dst_format') or sargs.get('img_format') or ''
            key = f'{base} {fmt}'.strip()
            f.cat_ns['blit, copy, clear'] += dur
            f.tr_ns[key] += dur
            f.tr_n[key] += 1
            continue
        if base == 'cmd_buffer_annotation':
            f.ann_ns[parse_kv(payload).get('str', '?')] += dur
            continue
        f.cat_ns['other: ' + base] += dur

    info = {'ctx': ctx, 'contexts': per_ctx_lines, 'bad': bad,
            'unmatched': unmatched + len(stack), 'app': app}
    return frames, info


def pick_frames(frames, spec):
    nrs = sorted(n for n, f in frames.items() if f.cb_spans)
    if not nrs:
        sys.exit('no complete frames (no cmd_buffer ranges): is the '
                 'cmd_buffer tracepoint switched off?')
    if spec:
        a, _, b = spec.partition('-')
        lo, hi = int(a), int(b or a)
        sel = [n for n in nrs if lo <= n <= hi]
        if not sel:
            sys.exit(f'no frames in {spec}; the trace has {nrs[0]}-{nrs[-1]}')
        return sel, 'chosen with --frames'
    rps = sorted(frames[n].n_rp for n in nrs)
    typical = rps[int(len(rps) * 0.9)]
    heavy = set(n for n in nrs if frames[n].n_rp >= max(1, typical / 2))
    best, cur = [], []
    for n in nrs:
        if n in heavy:
            cur.append(n)
        elif cur:
            best, cur = max(best, cur, key=len), []
    best = max(best, cur, key=len)
    # drop the edges: the first frames after loading still stream, the last
    # ones may already be the pause menu
    cut = len(best) // 20
    sel = best[cut:len(best) - cut] if len(best) > 40 else best
    return sel, (f'heavy frames: {len(heavy)} of {len(nrs)} have >= '
                 f'{max(1, typical // 2)} render passes; longest run, edges trimmed')


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


class Summary:
    """Per-frame averages over the selected frames of one trace."""

    def __init__(self, frames, sel):
        self.n = len(sel)
        fs = [frames[n] for n in sel]
        self.gpu_ms = [f.cb_ns / 1e6 for f in fs]
        starts = [f.first_ts for f in fs]
        gaps = [b - a for a, b in zip(starts, starts[1:]) if b > a]
        self.interval_ms = statistics.median(gaps) / 1e6 if gaps else None
        self.rp_per_frame = statistics.mean(f.n_rp for f in fs)
        self.disp_per_frame = statistics.mean(f.n_disp for f in fs)
        idle_in = 0
        for f in fs:
            spans = sorted(f.cb_spans)
            end = spans[0][1]
            for s, e in spans[1:]:
                if s > end:
                    idle_in += s - end
                end = max(end, e)
        self.idle_in_ms = idle_in / 1e6 / self.n

        def avg(attr):
            tot = collections.Counter()
            for f in fs:
                tot.update(getattr(f, attr))
            return tot

        self.cat = avg('cat_ns')
        self.rp_ns, self.rp_n = avg('rp_ns'), avg('rp_n')
        self.rp_draws, self.rp_gmem = avg('rp_draws'), avg('rp_gmem')
        self.cs_ns, self.cs_n, self.cs_groups = avg('cs_ns'), avg('cs_n'), avg('cs_groups')
        self.tr_ns, self.tr_n = avg('tr_ns'), avg('tr_n')
        self.ann_ns, self.in_rp_ns = avg('ann_ns'), avg('in_rp_ns')
        self.lrz_off, self.lrzw_off = avg('lrz_off_ns'), avg('lrzw_off_ns')
        self.mode_reason = avg('mode_reason_n')

    def per_frame_ms(self, ns):
        return ns / 1e6 / self.n


def timeline(frames, sel_set):
    """One line per second of GPU time: what the trace holds where."""
    nrs = sorted(n for n, f in frames.items() if f.first_ts is not None)
    if not nrs:
        return
    t0 = frames[nrs[0]].first_ts
    buckets = collections.OrderedDict()
    for n in nrs:
        f = frames[n]
        sec = int((f.first_ts - t0) / 1e9)
        b = buckets.setdefault(sec, [0, 0, 0.0, 0, n, n, 0])
        b[0] += 1
        b[1] += f.n_rp
        b[2] += f.cb_ns / 1e6
        b[3] += f.n_disp
        b[4] = min(b[4], n)
        b[5] = max(b[5], n)
        b[6] += n in sel_set
    print('\nTimeline (one line per second):')
    print('   s  frames   frame nr      rp/frame  disp/frame  GPU ms/frame  used')
    for sec, (cnt, rp, gms, disp, lo, hi, used) in buckets.items():
        mark = 'all' if used == cnt else (f'{used}' if used else '')
        print(f'{sec:4d}  {cnt:6d}  {lo:6d}-{hi:<6d}  {rp / cnt:8.1f}  {disp / cnt:10.1f}'
              f'  {gms / cnt:12.2f}  {mark}')


def fmt_ms(v):
    return f'{v:7.3f}'


def table(title, rows, cols, top, b_rows=None):
    """rows: {key: (ms_per_frame, extra columns...)}; sorted by ms."""
    keys = sorted(rows, key=lambda k: -rows[k][0])[:top]
    if b_rows is not None:
        for k in sorted(b_rows, key=lambda k: -b_rows[k][0])[:top]:
            if k not in keys:
                keys.append(k)
    print(f'\n{title}')
    if b_rows is None:
        print('   ms/frame  ' + '  '.join(cols))
    else:
        print('   A ms/fr    B ms/fr   change  ' + '  '.join(cols))
    for k in keys:
        a = rows.get(k)
        if b_rows is None:
            print(f'   {fmt_ms(a[0])}  ' + '  '.join(str(x) for x in a[1:]) + f'  {k}')
        else:
            b = b_rows.get(k)
            av = a[0] if a else 0.0
            bv = b[0] if b else 0.0
            ch = f'{(bv - av) / av * 100:+6.1f}%' if av > 0 and b else ('   new ' if not a else '  gone ')
            extra = (a or b)[1:]
            print(f'   {fmt_ms(av)}   {fmt_ms(bv)}  {ch}  ' + '  '.join(str(x) for x in extra) + f'  {k}')


def rows_rp(s):
    out = {}
    for k, ns in s.rp_ns.items():
        n = s.rp_n[k]
        out[k] = (s.per_frame_ms(ns), f'{n / s.n:5.1f}/fr', f'{ns / n / 1e3:8.1f} us',
                  f'{s.rp_draws[k] / n:6.0f} draws', f'gmem {s.rp_gmem[k] / n * 100:3.0f}%')
    return out


def rows_cs(s):
    out = {}
    for k, ns in s.cs_ns.items():
        n = s.cs_n[k]
        g = s.cs_groups[k] / n
        out[k] = (s.per_frame_ms(ns), f'{n / s.n:5.1f}/fr', f'{ns / n / 1e3:8.1f} us',
                  f'{g:8.0f} groups' if g else '     ? groups')
    return out


def rows_simple(s, ns_map, n_map=None):
    out = {}
    for k, ns in ns_map.items():
        cols = [s.per_frame_ms(ns)]
        if n_map is not None:
            cols.append(f'{n_map[k] / s.n:5.1f}/fr')
        out[k] = tuple(cols)
    return out


def report(path, frames, info, sel, how, s, other=None, top=25):
    print(f'Trace: {path}')
    print(f"Application: {info['app']}; context {info['ctx']} of "
          f"{dict(info['contexts'])} (events per context)")
    if info['bad'] or info['unmatched']:
        print(f"Skipped lines: {info['bad']}, unmatched ranges: {info['unmatched']}")
    print(f'Frames used: {sel[0]}-{sel[-1]} ({len(sel)} frames), {how}')
    if other is not None:
        p2, f2, i2, sel2, how2, s2 = other
        print(f'Compared with B: {p2}, frames {sel2[0]}-{sel2[-1]} ({len(sel2)}), {how2}')

    def frame_block(s):
        g = s.gpu_ms
        line = (f'GPU time per frame: mean {statistics.mean(g):.2f} ms, median {pct(g, .5):.2f}, '
                f'p90 {pct(g, .9):.2f}, p99 {pct(g, .99):.2f}, max {max(g):.2f}')
        if s.interval_ms:
            line += (f'\nFrame interval (median): {s.interval_ms:.2f} ms = {1000 / s.interval_ms:.1f} FPS; '
                     f'GPU busy {statistics.mean(g) / s.interval_ms * 100:.0f}% of it')
        line += (f'\nGPU idle between command buffers inside a frame: {s.idle_in_ms:.2f} ms/frame'
                 f'\nRender passes {s.rp_per_frame:.1f}/frame, dispatches {s.disp_per_frame:.1f}/frame')
        return line

    print('\n[A] ' + frame_block(s).replace('\n', '\n    '))
    if other is not None:
        print('[B] ' + frame_block(s2).replace('\n', '\n    '))
        a, b = statistics.mean(s.gpu_ms), statistics.mean(s2.gpu_ms)
        print(f'B against A: GPU time per frame {(b - a) / a * 100:+.1f}%')

    total = sum(s.cat.values())
    cats = {k: (s.per_frame_ms(v), f'{v / total * 100:4.1f}%') for k, v in s.cat.items()}
    table('Where the GPU time goes (top-level ranges):', cats, ['share'], top,
          None if other is None else {k: (s2.per_frame_ms(v), '') for k, v in s2.cat.items()})

    table(f'Render passes by size (top {top}):', rows_rp(s),
          ['count', 'avg', 'draws', 'mode'], top, None if other is None else rows_rp(s2))
    table(f'Compute shaders (top {top}; sha prefix, workgroup size):', rows_cs(s),
          ['count', 'avg', 'groups'], top, None if other is None else rows_cs(s2))
    if s.tr_ns:
        table('Blits, copies, clears:', rows_simple(s, s.tr_ns, s.tr_n), ['count'], top,
              None if other is None else rows_simple(s2, s2.tr_ns, s2.tr_n))
    if s.in_rp_ns:
        table('Inside render passes (already counted in them):', rows_simple(s, s.in_rp_ns), [],
              top, None if other is None else rows_simple(s2, s2.in_rp_ns))
    if s.lrz_off:
        table('Render-pass time with LRZ off, by reason:', rows_simple(s, s.lrz_off), [], top)
    if s.lrzw_off:
        table('Render-pass time with LRZ writes off, by reason:', rows_simple(s, s.lrzw_off), [], top)
    if s.mode_reason:
        print('\nForced render mode, passes per frame by reason:')
        for k, v in s.mode_reason.most_common(top):
            print(f'   {v / s.n:6.2f}  {k}')
    if s.ann_ns:
        table('Debug labels from the game:', rows_simple(s, s.ann_ns), [], top,
              None if other is None else rows_simple(s2, s2.ann_ns))


def anatomy(path, ctx, frame_nr):
    """Print one frame's top-level ranges in GPU order."""
    print(f'\nFrame {frame_nr} in GPU order (top-level ranges; repeats of the same dispatch merged):')
    stack = []
    rows = []
    for line in open_lines(path):
        if not line or line[0] not in '0123456789':
            continue
        parts = line.rstrip('\n').split(',', 4)
        if len(parts) < 5:
            continue
        rest = parts[4]
        i = rest.rfind('ctx=')
        if i < 0 or int(rest[i + 4:]) != ctx:
            continue
        name, ts, payload = parts[3], int(parts[2]), rest[:i].rstrip(', ')
        if name.startswith('start_'):
            stack.append((name[6:], ts, int(parts[0]), payload))
        elif name.startswith('end_') and stack:
            base = name[4:]
            j = len(stack) - 1
            while j >= 0 and stack[j][0] != base:
                j -= 1
            if j < 0:
                continue
            sbase, sts, sframe, spayload = stack[j]
            depth = j
            del stack[j:]
            if sframe != frame_nr or base in IN_RENDER_PASS:
                continue
            if base == 'cmd_buffer':
                rows.append(['cmd buffer', sts, ts - sts, 1, depth])
                continue
            a = parse_kv(spayload)
            if base == 'render_pass':
                e = parse_kv(payload)
                desc = (f"render pass {rp_signature(a)} "
                        f"{'gmem' if e.get('tiledRender') == 'true' else 'sysmem'} "
                        f"draws={e.get('drawCount', '?')}")
            elif base in ('compute', 'compute_indirect'):
                desc = f'dispatch {cs_signature(base, a)}'
            elif base == 'cmd_buffer_annotation':
                desc = 'label ' + parse_kv(payload).get('str', '?')
            else:
                desc = base
            if rows and rows[-1][0] == desc and desc.startswith('dispatch'):
                rows[-1][2] += ts - sts
                rows[-1][3] += 1
            else:
                rows.append([desc, sts, ts - sts, 1, depth])
    if not rows:
        print('   (nothing)')
        return
    t0 = min(r[1] for r in rows)
    for desc, sts, dur, n, depth in sorted(rows, key=lambda r: (r[1], r[4])):
        times = f' x{n}' if n > 1 else ''
        print(f'   {(sts - t0) / 1e3:9.1f} us  {dur / 1e3:8.1f} us  {"  " * min(depth, 1)}{desc}{times}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('trace')
    ap.add_argument('--compare', metavar='TRACE2')
    ap.add_argument('--frames', metavar='A-B', help='frame range of TRACE')
    ap.add_argument('--frames2', metavar='A-B', help='frame range of TRACE2')
    ap.add_argument('--ctx', type=int, help='context id (default: the busiest)')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--anatomy', action='store_true',
                    help="also list the median frame's ranges in GPU order")
    args = ap.parse_args()

    frames, info = load(args.trace, args.ctx)
    sel, how = pick_frames(frames, args.frames)
    timeline(frames, set(sel))
    s = Summary(frames, sel)
    other = None
    if args.compare:
        f2, i2 = load(args.compare, args.ctx)
        sel2, how2 = pick_frames(f2, args.frames2)
        other = (args.compare, f2, i2, sel2, how2, Summary(f2, sel2))
    print()
    report(args.trace, frames, info, sel, how, s, other, args.top)
    if args.anatomy:
        med = sorted(sel, key=lambda n: frames[n].cb_ns)[len(sel) // 2]
        anatomy(args.trace, info['ctx'], med)


if __name__ == '__main__':
    main()
