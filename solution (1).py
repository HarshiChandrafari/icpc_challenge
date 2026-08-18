import sys
from collections import deque
from heapq import heappush, heappop
from bisect import bisect_left


def main():
    fin = sys.stdin
    wr = sys.stdout.write
    flush = sys.stdout.flush

    def rtok(n):
        t = fin.readline().split()
        while not t:
            s = fin.readline()
            if not s:
                sys.exit(0)
            t = s.split()
        while len(t) < n:
            s = fin.readline()
            if not s:
                sys.exit(0)
            t += s.split()
        return t

    p = rtok(6)
    K = int(p[0]); S = float(p[1]); LAT = float(p[2]); BW = float(p[3])
    BPT = int(p[4]); NL = int(p[5])
    p = rtok(7)
    SLO1 = float(p[0]); SLO2 = float(p[1])
    TP_UB = float(p[2]); TP_BASE = float(p[3]); DIST_BASE = float(p[4])
    W_TP = float(p[5]); W_C = float(p[6])

    NROWS = int(rtok(1)[0])
    rows = [rtok(7) for _ in range(NROWS)]
    rows.sort(key=lambda a: int(a[0]))
    xs = [[] for _ in range(6)]
    ys = [[] for _ in range(6)]
    for t in rows:
        b = int(t[0])
        for c in range(6):
            v = float(t[1 + c])
            if v >= 0.0:
                xs[c].append(b); ys[c].append(v)
    cache = [{} for _ in range(6)]

    def interp(c, b):
        d = cache[c]
        v = d.get(b)
        if v is not None:
            return v
        X = xs[c]; Y = ys[c]
        if not X:
            r = 1.0
        elif b <= X[0]:
            r = Y[0]
        elif b >= X[-1]:
            r = Y[-1]
        else:
            i = bisect_left(X, b)
            if X[i] == b:
                r = Y[i]
            else:
                x0 = X[i-1]; x1 = X[i]; y0 = Y[i-1]; y1 = Y[i]
                r = y0 + (y1-y0)*(b-x0)/(x1-x0)
        d[b] = r
        return r

    DP1 = interp(4, 1)

    def local_cost_per_req(b):
        return (2.0*S + interp(3, b) + interp(5, b)) / b
    BMAX = 512
    ref = local_cost_per_req(BMAX)
    BATCH_TARGET = BMAX
    for b in range(1, BMAX+1):
        if local_cost_per_req(b) <= ref*1.10:
            BATCH_TARGET = b
            break
    if BATCH_TARGET > 256:
        BATCH_TARGET = 256

    # ---- static mode from the scoring weights ----
    # dist_base == 0 makes the waiting component all-or-nothing: it is 1 only at
    # dist == 0. Once we are provably above it, latency can no longer earn points.
    HARD_SLO = (DIST_BASE <= 0.0)
    ADMIT_CAP = 1 << 30 if W_TP >= W_C else BATCH_TARGET

    ST = []; LIN = []; ASG = []; CUR = []; LTOK = []; PFULL = []; ARRT = []; TOKS = []

    ready_ppre = []; ready_ppost = []
    ready_pproc = [[] for _ in range(K)]
    ready_dproc = [set() for _ in range(K)]
    ready_dpre = set(); ready_dpost = set()

    local_busy = False
    rem_busy = [False]*K
    task_local = None
    task_rem = [None]*K
    rem_pref_ms = [0.0]*K
    dec_active = [0]*K

    # ---- live scoring telemetry (all O(1) to maintain) ----
    tdr_sum = 0.0; tdr_n = 0
    tpot_sum = 0.0; tpot_n = 0
    n_pend = 0; sum_arr_pend = 0.0      # requests not yet decode-ready
    n_dec = 0; sum_ltok = 0.0           # requests in the decode loop
    n_waitdown = 0                      # decode results in flight toward the edge
    tokens = 0; fin_cnt = 0; multi_cnt = 0
    first_arr = -1.0

    def pick_remote():
        best = 0; bv = 1e300
        for k in range(K):
            v = rem_pref_ms[k] + dec_active[k]*DP1*20.0
            if v < bv:
                bv = v; best = k
        return best

    while True:
        line = fin.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        if line[0] == 'E':
            return
        now = float(line)
        e = int(fin.readline())
        fins = []

        for _ in range(e):
            t = fin.readline().split()
            typ = t[0]

            if typ == 'TDN':
                srv = t[1]
                if srv == 'E':
                    tk = task_local; task_local = None; local_busy = False
                    kind = tk[0]
                    if kind == 'PPRE':
                        ST[tk[1]] = 2
                    elif kind == 'PPOST':
                        r = tk[1]
                        ST[r] = 8
                        tdr_sum += now - ARRT[r]; tdr_n += 1
                        n_pend -= 1; sum_arr_pend -= ARRT[r]
                        n_dec += 1; LTOK[r] = now; sum_ltok += now
                        dec_active[ASG[r]] += 1
                        ready_dpre.add(r)
                    elif kind == 'DPRE':
                        for r in tk[1]:
                            ST[r] = 10
                    else:
                        for r in tk[1]:
                            ST[r] = 8
                            tokens += 1
                            if TOKS[r] >= 1:
                                tpot_sum += now - LTOK[r]; tpot_n += 1
                                if TOKS[r] == 1:
                                    multi_cnt += 1
                            TOKS[r] += 1
                            sum_ltok += now - LTOK[r]; LTOK[r] = now
                            ready_dpre.add(r)
                else:
                    k = int(srv[1:])
                    tk = task_rem[k]; task_rem[k] = None; rem_busy[k] = False
                    if tk[0] == 'PPROC':
                        r = tk[1]; ls = tk[2]; le = tk[3]
                        rem_pref_ms[k] -= (le-ls)/NL*PFULL[r]
                        CUR[r] = le
                        if le < NL:
                            ST[r] = 3; heappush(ready_pproc[k], (PFULL[r], r))
                        else:
                            ST[r] = 5
                    else:
                        for r in tk[1]:
                            ST[r] = 13
                        n_waitdown += len(tk[1])

            elif typ == 'XDN':
                k = int(t[2]); m = int(t[5])
                if t[4] == 'PRE':
                    r = int(t[6])
                    if t[1] == 'UP':
                        ST[r] = 3; heappush(ready_pproc[k], (PFULL[r], r))
                    else:
                        ST[r] = 6; heappush(ready_ppost, (interp(2, LIN[r]), r))
                else:
                    if t[1] == 'UP':
                        s = ready_dproc[k]
                        for x in t[6:6+m]:
                            r = int(x); ST[r] = 11; s.add(r)
                    else:
                        for x in t[6:6+m]:
                            r = int(x); ST[r] = 14; ready_dpost.add(r)
                        n_waitdown -= m

            elif typ == 'ARR':
                rid = int(t[1]); lin = int(t[2])
                while len(ST) <= rid:
                    ST.append(0); LIN.append(0); ASG.append(0); CUR.append(0)
                    LTOK.append(0.0); PFULL.append(0.0); ARRT.append(0.0); TOKS.append(0)
                ST[rid] = 0; LIN[rid] = lin; CUR[rid] = 0; TOKS[rid] = 0
                ARRT[rid] = now; PFULL[rid] = interp(1, lin)
                if first_arr < 0.0:
                    first_arr = now
                n_pend += 1; sum_arr_pend += now
                heappush(ready_ppre, (PFULL[rid], rid))
            else:
                fins.append(int(t[1]))

        for rid in fins:
            ST[rid] = 16
            ready_dpre.discard(rid)
            dec_active[ASG[rid]] -= 1
            n_dec -= 1; sum_ltok -= LTOK[rid]
            fin_cnt += 1

        # ---------- live scoring state ----------
        # Projected means include the waiting time already accrued by requests
        # still in flight, so pressure builds before the metric is realised.
        den = tdr_n + n_pend
        m_tdr = (tdr_sum + n_pend*now - sum_arr_pend)/den if den else 0.0
        # If every finished request produced a single token there are no gaps at
        # all, so TPOT is structurally zero and must not steer anything.
        if fin_cnt >= 8 and multi_cnt == 0:
            m_tpot = 0.0
        else:
            den2 = tpot_n + n_dec
            m_tpot = (tpot_sum + n_dec*now - sum_ltok)/den2 if den2 else 0.0

        r_tdr = m_tdr/SLO1
        r_tpot = m_tpot/SLO2
        ex1 = r_tdr - 1.0
        if ex1 < 0.0: ex1 = 0.0
        ex2 = r_tpot - 1.0
        if ex2 < 0.0: ex2 = 0.0
        dist = (ex1*ex1 + ex2*ex2) ** 0.5

        # Can either component still earn points?
        if W_C <= 0.0:
            lat_live = False
        elif HARD_SLO:
            lat_live = (dist <= 0.0)
        else:
            lat_live = (dist < DIST_BASE*1.5)
        thr_live = (W_TP > 0.0)
        if thr_live and first_arr >= 0.0 and now > first_arr:
            thr_live = (tokens/(now - first_arr)) < TP_UB

        # Throughput mode: latency is either worthless or has slack to spare.
        TP_MODE = thr_live and ((not lat_live) or W_TP >= 0.7)

        if not lat_live:
            prefer_pref = True
        elif not thr_live:
            prefer_pref = (r_tdr >= r_tpot)
        elif W_TP >= W_C:
            prefer_pref = (r_tdr >= r_tpot) or (n_dec < ADMIT_CAP)
        else:
            prefer_pref = (r_tdr >= r_tpot)

        # Merging D POSTs saves a whole edge task. Deferring is only worthwhile
        # when the wait would at least double the group, and it never idles the
        # edge because D POST stays in the fallback order.
        defer_dpost = TP_MODE and n_waitdown > 0 and n_waitdown >= len(ready_dpost)

        res = []
        if not local_busy:
            if prefer_pref:
                order = ('P', 'D', 'X') if defer_dpost else ('X', 'P', 'D')
            else:
                order = ('D', 'P', 'X') if defer_dpost else ('X', 'D', 'P')
            for kind in order:
                if kind == 'X':
                    if ready_dpost:
                        g = list(ready_dpost); ready_dpost.clear()
                        for r in g: ST[r] = 15
                        task_local = ('DPOST', g); local_busy = True
                        res.append('E D POST -1 %d %s' % (len(g), ' '.join(map(str, g))))
                        break
                elif kind == 'P':
                    if ready_ppost:
                        r = heappop(ready_ppost)[1]; ST[r] = 7
                        task_local = ('PPOST', r); local_busy = True
                        res.append('E P POST %d %d' % (ASG[r], r)); break
                    if ready_ppre:
                        r = heappop(ready_ppre)[1]; k = pick_remote()
                        ASG[r] = k; ST[r] = 1
                        rem_pref_ms[k] += PFULL[r]
                        task_local = ('PPRE', r); local_busy = True
                        res.append('E P PRE %d %d' % (k, r)); break
                else:
                    if ready_dpre:
                        g = list(ready_dpre); ready_dpre.clear()
                        for r in g: ST[r] = 9
                        task_local = ('DPRE', g); local_busy = True
                        res.append('E D PRE -1 %d %s' % (len(g), ' '.join(map(str, g))))
                        break

        for k in range(K):
            if rem_busy[k]:
                continue
            dq = ready_dproc[k]; pq = ready_pproc[k]
            if dq and pq:
                use_pref = prefer_pref
            elif dq:
                use_pref = False
            elif pq:
                use_pref = True
            else:
                continue
            if use_pref:
                r = heappop(pq)[1]; ST[r] = 4
                ls = CUR[r]
                task_rem[k] = ('PPROC', r, ls, NL); rem_busy[k] = True
                res.append('C%d P PROC %d %d %d %d' % (k, ls, NL, k, r))
            else:
                g = list(dq); dq.clear()
                for r in g: ST[r] = 12
                task_rem[k] = ('DPROC', g); rem_busy[k] = True
                res.append('C%d D PROC %d %d %s' % (k, k, len(g), ' '.join(map(str, g))))

        if res:
            wr('%d\n%s\n' % (len(res), '\n'.join(res)))
        else:
            wr('0\n')
        flush()


main()
