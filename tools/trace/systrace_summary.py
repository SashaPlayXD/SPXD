#!/usr/bin/env python3
"""
Summarize an Android system trace for one game process: which threads burn
CPU, what they wait on, how fast each core ran, and the GPU clock.

usage: systrace_summary.py TRACE [--process REGEX] [--from S] [--to S]
                           [--top N]

TRACE is the Perfetto file from Developer options -> System Tracing (also
inside a .zip, or gzipped). Needs the Perfetto trace protos from PyPI:

  pip install perfetto protobuf

Only perfetto_trace_pb2 is imported; the trace_processor binary that the
perfetto package can download is not used.

What it reads: sched_switch and sched_waking (plain or compact_sched),
cpu_frequency, gpu_frequency / kgsl_gpu_frequency, process_tree, task_newtask
and task_rename, sys_stats cpufreq samples, and atrace "queueBuffer" slices
for the on-screen frame rate.

For each thread of the game it splits the window into running, runnable
(ready but waiting for a CPU, including after a wakeup), sleeping (S) and
uninterruptible (D), and names who woke it. A thread that runs most of the
window is the bottleneck; one that sleeps a lot and is woken by a GPU
interrupt or a kgsl thread is waiting for the GPU.

--process picks the game by a regex on the process command line or the
thread names (default: Cyberpunk, else *.exe, else the busiest process).
--from/--to cut a window in seconds from the start of the trace.
"""
import argparse
import bisect
import collections
import gzip
import io
import re
import sys
import zipfile
import zlib

try:
    from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace
except ImportError:
    sys.exit('needs the Perfetto protos: pip install perfetto protobuf')

CLONE_THREAD = 0x10000


def read_trace_bytes(path):
    data = open(path, 'rb').read()
    if data[:4] == b'PK\x03\x04':
        z = zipfile.ZipFile(io.BytesIO(data))
        name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
        print(f'note: using {name} from the zip', file=sys.stderr)
        data = z.read(name)
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    return data


def packets(data):
    tr = Trace.FromString(data)
    for p in tr.packet:
        if p.HasField('compressed_packets'):
            raw = p.compressed_packets
            try:
                inner = zlib.decompress(raw)
            except zlib.error:
                inner = zlib.decompress(raw, -15)
            yield from packets(inner)
        else:
            yield p


def state_letter(v):
    if v & 0xff == 0:
        return 'R'  # preempted (TASK_REPORT_MAX bit) or yielded
    if v & 0x2:
        return 'D'
    if v & 0x1:
        return 'S'
    if v & 0x80:
        return 'I'
    return 'X'


class Model:
    def __init__(self):
        self.comm = {}       # tid -> thread name
        self.tgid = {}       # tid -> process id
        self.cmdline = {}    # pid -> command line
        self.switches = collections.defaultdict(list)  # cpu -> (ts, prev, state, next)
        self.wakings = []    # (ts, tid, waker or None, cpu of the event)
        self.wakeups = []
        self.freq = collections.defaultdict(list)       # cpu -> (ts, khz)
        self.gpu_freq = []   # (ts, khz)
        self.queue_buffer = collections.Counter()      # pid -> count
        self.qb_ts = collections.defaultdict(list)
        self.kgsl_events = collections.Counter()
        self.t_min = None
        self.t_max = None

    def seen(self, ts):
        if self.t_min is None or ts < self.t_min:
            self.t_min = ts
        if self.t_max is None or ts > self.t_max:
            self.t_max = ts


def load(path):
    m = Model()
    for p in packets(read_trace_bytes(path)):
        which = p.WhichOneof('data')
        if which == 'process_tree':
            for pr in p.process_tree.processes:
                if pr.cmdline:
                    m.cmdline[pr.pid] = ' '.join(pr.cmdline)
                m.tgid.setdefault(pr.pid, pr.pid)
            for th in p.process_tree.threads:
                m.tgid[th.tid] = th.tgid
                if th.name:
                    m.comm.setdefault(th.tid, th.name)
        elif which == 'sys_stats':
            for cpu, khz in enumerate(p.sys_stats.cpufreq_khz):
                m.freq[cpu].append((p.timestamp, khz))
        elif which == 'ftrace_events':
            b = p.ftrace_events
            cpu = b.cpu
            if b.HasField('compact_sched'):
                cs = b.compact_sched
                t = 0
                for i, d in enumerate(cs.switch_timestamp):
                    t += d
                    nxt = cs.switch_next_pid[i]
                    m.switches[cpu].append((t, None, cs.switch_prev_state[i], nxt))
                    if nxt and i < len(cs.switch_next_comm_index):
                        m.comm[nxt] = cs.intern_table[cs.switch_next_comm_index[i]]
                    m.seen(t)
                t = 0
                for i, d in enumerate(cs.waking_timestamp):
                    t += d
                    tid = cs.waking_pid[i]
                    m.wakings.append((t, tid, None, cpu))
                    if i < len(cs.waking_comm_index):
                        m.comm.setdefault(tid, cs.intern_table[cs.waking_comm_index[i]])
            for e in b.event:
                ts = e.timestamp
                kind = e.WhichOneof('event')
                if kind == 'sched_switch':
                    s = e.sched_switch
                    m.switches[cpu].append((ts, s.prev_pid, s.prev_state, s.next_pid))
                    if s.next_pid:
                        m.comm[s.next_pid] = s.next_comm
                    if s.prev_pid:
                        m.comm[s.prev_pid] = s.prev_comm
                    m.seen(ts)
                elif kind == 'sched_waking':
                    w = e.sched_waking
                    m.wakings.append((ts, w.pid, e.pid, cpu))
                    m.comm.setdefault(w.pid, w.comm)
                elif kind == 'sched_wakeup':
                    w = e.sched_wakeup
                    m.wakeups.append((ts, w.pid, None, cpu))
                elif kind == 'cpu_frequency':
                    f = e.cpu_frequency
                    m.freq[f.cpu_id].append((ts, f.state))
                elif kind == 'gpu_frequency':
                    m.gpu_freq.append((ts, e.gpu_frequency.state))
                elif kind == 'kgsl_gpu_frequency':
                    m.gpu_freq.append((ts, e.kgsl_gpu_frequency.gpu_freq))
                elif kind == 'task_newtask':
                    t = e.task_newtask
                    m.comm[t.pid] = t.comm
                    if t.clone_flags & CLONE_THREAD:
                        m.tgid[t.pid] = m.tgid.get(e.pid, e.pid)
                    else:
                        m.tgid[t.pid] = t.pid
                elif kind == 'task_rename':
                    m.comm[e.task_rename.pid] = e.task_rename.newcomm
                elif kind == 'print':
                    buf = e.print.buf
                    if buf.startswith('B|') and 'queueBuffer' in buf:
                        parts = buf.split('|', 2)
                        try:
                            pid = int(parts[1])
                        except (IndexError, ValueError):
                            continue
                        if parts[2].strip().startswith('queueBuffer'):
                            m.queue_buffer[pid] += 1
                            m.qb_ts[pid].append(ts)
                elif kind and kind.startswith('kgsl'):
                    m.kgsl_events[kind] += 1
    for cpu in m.switches:
        m.switches[cpu].sort(key=lambda x: x[0])
    for cpu in m.freq:
        m.freq[cpu].sort()
    m.gpu_freq.sort()
    m.wakings.sort()
    m.wakeups.sort()
    return m


def build_slices(m, lo, hi):
    """Running intervals clipped to [lo, hi): tid -> [(s, e, cpu, end_state)],
    and per cpu sorted (start, tid) for 'who ran here at time t'."""
    by_tid = collections.defaultdict(list)
    per_cpu = {}
    for cpu, sw in m.switches.items():
        starts, tids = [], []
        cur = None
        for i, (ts, prev, state, nxt) in enumerate(sw):
            if prev is None:
                prev = cur
            if cur is not None and i > 0:
                s0 = sw[i - 1][0]
                s, e = max(s0, lo), min(ts, hi)
                if e > s:
                    by_tid[cur].append((s, e, cpu, state_letter(state) if prev == cur else '?'))
            starts.append(ts)
            tids.append(nxt)
            cur = nxt
        if sw and cur is not None:
            s, e = max(sw[-1][0], lo), hi
            if e > s:
                by_tid[cur].append((s, e, cpu, None))
        per_cpu[cpu] = (starts, tids)
    for v in by_tid.values():
        v.sort()
    return by_tid, per_cpu


def running_on(per_cpu, cpu, ts):
    if cpu not in per_cpu:
        return None
    starts, tids = per_cpu[cpu]
    i = bisect.bisect_right(starts, ts) - 1
    return tids[i] if i >= 0 else None


def freq_integral(fr, s, e):
    """Integral of the frequency step function fr [(ts, khz)] over [s, e):
    returns (khz*ns, ns covered)."""
    if not fr:
        return 0, 0
    i = bisect.bisect_right(fr, (s, float('inf'))) - 1
    total, covered = 0, 0
    t = s
    if i < 0:
        i = 0
        t = max(s, fr[0][0])
    while t < e and i < len(fr):
        nxt = fr[i + 1][0] if i + 1 < len(fr) else e
        seg_e = min(nxt, e)
        if seg_e > t:
            total += fr[i][1] * (seg_e - t)
            covered += seg_e - t
        t = max(t, seg_e)
        i += 1
    return total, covered


def name_of(m, tid):
    if tid == 0:
        return 'idle/irq (pid 0)'
    return f"{m.comm.get(tid, '?')} [{tid}]"


def proc_name(m, pid):
    c = m.cmdline.get(pid)
    if c:
        return c[:60]
    return m.comm.get(pid, '?')


def pick_process(m, cpu_by_pid, regex):
    pats = [regex] if regex else [r'(?i)cyberpunk', r'(?i)\.exe']
    for pat in pats:
        cands = []
        for pid, ns in cpu_by_pid.items():
            names = [m.cmdline.get(pid, ''), m.comm.get(pid, '')]
            names += [m.comm.get(t, '') for t, g in m.tgid.items() if g == pid][:200]
            if any(re.search(pat, n) for n in names if n):
                cands.append((ns, pid))
        if cands:
            return max(cands)[1], pat
    if regex:
        sys.exit(f'no process matches {regex}')
    return max(cpu_by_pid.items(), key=lambda kv: kv[1])[0], 'busiest process'


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('trace')
    ap.add_argument('--process', help='regex for the game process')
    ap.add_argument('--from', dest='t_from', type=float, default=0.0)
    ap.add_argument('--to', dest='t_to', type=float)
    ap.add_argument('--top', type=int, default=15)
    args = ap.parse_args()

    m = load(args.trace)
    if m.t_min is None:
        sys.exit('no sched_switch events: was "CPU Scheduling" recorded?')
    lo = m.t_min + int(args.t_from * 1e9)
    hi = m.t_max if args.t_to is None else min(m.t_max, m.t_min + int(args.t_to * 1e9))
    wall = hi - lo
    by_tid, per_cpu = build_slices(m, lo, hi)
    cpus = sorted(m.switches)

    def tg(tid):
        return m.tgid.get(tid, tid)

    run_by_tid = {t: sum(e - s for s, e, _, _ in v) for t, v in by_tid.items() if t}
    cpu_by_pid = collections.Counter()
    for t, ns in run_by_tid.items():
        cpu_by_pid[tg(t)] += ns

    print(f'Trace: {args.trace}')
    print(f'Window: {(lo - m.t_min) / 1e9:.1f}-{(hi - m.t_min) / 1e9:.1f} s of '
          f'{(m.t_max - m.t_min) / 1e9:.1f} s, {len(cpus)} CPUs, '
          f'{sum(len(v) for v in m.switches.values())} context switches, '
          f'{len(m.wakings)} wakeups recorded')

    print('\nCPU time by process (1.00 = one core busy the whole window):')
    for pid, ns in cpu_by_pid.most_common(args.top):
        print(f'   {ns / wall:5.2f}  {proc_name(m, pid)} [{pid}]')

    game, how = pick_process(m, cpu_by_pid, args.process)
    threads = sorted((t for t in run_by_tid if tg(t) == game), key=lambda t: -run_by_tid[t])
    print(f'\nGame process: {proc_name(m, game)} [{game}] ({how}), {len(threads)} threads ran, '
          f'{cpu_by_pid[game] / wall:.2f} cores in total')

    # per-thread states: running from slices; after a slice the thread is
    # runnable (R end state) or asleep (S/D) until a wakeup, then runnable.
    wakes = m.wakings if m.wakings else m.wakeups
    wake_ts = collections.defaultdict(list)
    for ts, tid, waker, cpu in wakes:
        wake_ts[tid].append((ts, waker, cpu))

    print(f'\nGame threads (top {args.top} by CPU time), share of the window:')
    print('    run%  runnable%  sleep%  D%   MHz while running  CPUs                  wakes/s  woken by')
    worst = None
    for t in threads[:args.top]:
        sl = by_tid[t]
        run = run_by_tid[t]
        runnable = sleep = dsleep = 0
        wakers = collections.Counter()
        wl = wake_ts.get(t, [])
        wl_ts = [w[0] for w in wl]
        n_wakes = 0
        for (s, e, cpu, st), nxt in zip(sl, sl[1:] + [None]):
            gap_end = nxt[0] if nxt else hi
            if gap_end <= e:
                continue
            if st == 'R' or st is None:
                runnable += gap_end - e
                continue
            i = bisect.bisect_left(wl_ts, e)
            if i < len(wl) and wl_ts[i] < gap_end:
                ts_w, waker, wcpu = wl[i]
                if waker is None:
                    waker = running_on(per_cpu, wcpu, ts_w)
                wakers[waker] += 1
                n_wakes += 1
                asleep, ready = ts_w - e, gap_end - ts_w
            else:
                asleep, ready = gap_end - e, 0
            if st == 'D':
                dsleep += asleep
            else:
                sleep += asleep
            runnable += ready
        # time before the first slice in the window
        if sl and sl[0][0] > lo:
            sleep += sl[0][0] - lo
        khz_ns = covered = 0
        cpu_share = collections.Counter()
        for s, e, cpu, _ in sl:
            a, c = freq_integral(m.freq.get(cpu, []), s, e)
            khz_ns += a
            covered += c
            cpu_share[cpu] += e - s
        mhz = f'{khz_ns / covered / 1000:6.0f}' if covered else '     ?'
        cpus_s = ' '.join(f'{c}:{v / run * 100:.0f}%' for c, v in cpu_share.most_common(3))
        woken = ', '.join(f'{name_of(m, w) if w is not None else "?"} {n}'
                          for w, n in wakers.most_common(3))
        print(f'   {run / wall * 100:5.1f}  {runnable / wall * 100:8.1f}  {sleep / wall * 100:6.1f}'
              f'  {dsleep / wall * 100:4.1f}  {mhz:>17}  {cpus_s:<20}  {n_wakes / (wall / 1e9):7.0f}'
              f'  {woken}   <- {name_of(m, t)}')
        if worst is None:
            worst = (t, run / wall, sleep / wall, wakers)

    print('\nCPUs:')
    print('   cpu  busy%  avg MHz  max MHz seen  time at >=95% of max')
    maxes = {}
    for cpu in cpus:
        busy = sum(e - s for t, v in by_tid.items() if t for s, e, c, _ in v if c == cpu)
        fr = m.freq.get(cpu, [])
        a, c = freq_integral(fr, lo, hi)
        mx = max((k for _, k in fr), default=0)
        maxes[cpu] = mx
        top_ns = 0
        if mx:
            for i, (ts, khz) in enumerate(fr):
                nxt = fr[i + 1][0] if i + 1 < len(fr) else hi
                s, e = max(ts, lo), min(nxt, hi)
                if e > s and khz >= 0.95 * mx:
                    top_ns += e - s
        avg = f'{a / c / 1000:7.0f}' if c else '      ?'
        print(f'   {cpu:3d}  {busy / wall * 100:5.1f}  {avg}  {mx / 1000:12.0f}  '
              f'{(top_ns / c * 100 if c else 0):5.1f}%')

    if m.gpu_freq:
        fr = m.gpu_freq
        a, c = freq_integral(fr, lo, hi)
        dist = collections.Counter()
        for i, (ts, khz) in enumerate(fr):
            nxt = fr[i + 1][0] if i + 1 < len(fr) else hi
            s, e = max(ts, lo), min(nxt, hi)
            if e > s:
                dist[khz] += e - s
        unit = 1e6 if max(dist or [0]) > 1e7 else 1e3  # Hz or kHz
        print(f'\nGPU clock: average {a / c / unit:.0f} MHz' if c else '\nGPU clock: no samples in the window')
        for khz, ns in sorted(dist.items(), reverse=True):
            print(f'   {khz / unit:6.0f} MHz  {ns / wall * 100:5.1f}%')
    else:
        print('\nGPU clock: not in the trace')
    if m.kgsl_events:
        print('KGSL events: ' + ', '.join(f'{k} {v}' for k, v in m.kgsl_events.most_common()))

    if m.queue_buffer:
        print('\nFrames handed to the display (queueBuffer per second):')
        for pid, n in m.queue_buffer.most_common(5):
            ts = [t for t in m.qb_ts[pid] if lo <= t < hi]
            if len(ts) > 1:
                print(f'   {len(ts) / (wall / 1e9):6.1f}/s  {proc_name(m, pid)} [{pid}]')

    if worst:
        t, run, sleep, wakers = worst
        print('\nReading:')
        if run > 0.85:
            print(f'   {name_of(m, t)} runs {run * 100:.0f}% of the window: the game is bound by this '
                  f'one thread on the CPU.')
        else:
            top = wakers.most_common(1)
            by = name_of(m, top[0][0]) if top and top[0][0] is not None else 'unknown'
            print(f'   The busiest game thread {name_of(m, t)} runs only {run * 100:.0f}% and sleeps '
                  f'{sleep * 100:.0f}%; most often woken by {by}. If that is a kgsl/irq thread, it '
                  f'waits for the GPU; if a game thread, for its own workers.')


if __name__ == '__main__':
    main()
