// ============================================================================
//  Edge/Cloud Inference Scheduler  (C++ translation of the reference solution)
// ----------------------------------------------------------------------------
//  This program plays the "scheduler" side of an interactive judge that
//  simulates requests flowing through a two-tier system:
//
//      local computer (E)  <-- link -->  remote computers (C0 .. C{K-1})
//
//  Every request goes through six steps, in this order:
//
//      P PRE (E) -> [uplink]  -> P PROC (Ck) -> [downlink] -> P POST (E)
//         (then repeated once per output token:)
//      D PRE (E) -> [uplink]  -> D PROC (Ck) -> [downlink] -> D POST (E)
//
//  P PRE/P PROC/P POST run once per request ("prefill" / input stage).
//  D PRE/D PROC/D POST run once per *output token* ("decode" / output step).
//
//  The judge drives the interaction: it sends "frames" (a timestamp, an
//  event count, then that many events -- ARR/TDN/XDN/FIN), and after each
//  frame we print the tasks we want to start on any currently-free
//  computer. Transfers (uplink/downlink) are automatic; we never schedule
//  them ourselves, we only react to their XDN completion events.
//
//  This file is a direct, line-by-line port of the reference Python
//  scheduler: it keeps the same per-request state machine, the same
//  "ready" queues per pipeline stage, and the same heuristics for choosing
//  between latency-oriented and throughput-oriented scheduling. See the
//  inline comments for the reasoning behind each heuristic.
// ============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------------
// A small min-heap of (priority, request-id) pairs, matching Python's
// heapq usage on (float, int) tuples: the pair with the smallest priority
// comes out first, ties broken by the smaller request id.
// ---------------------------------------------------------------------------
using PriorityQueueD = priority_queue<pair<double, int>,
                                       vector<pair<double, int>>,
                                       greater<pair<double, int>>>;

// ---------------------------------------------------------------------------
// Task-time table: piecewise-linear interpolation of task durations, one
// curve per task type, built from the judge's warm-up table.
// ---------------------------------------------------------------------------
class TaskTimeTable {
public:
    // Column order matches the judge's 7-column table:
    //   batch_size  prefill_pre  prefill_proc  prefill_post
    //               decode_pre   decode_proc   decode_post
    enum Column {
        PREFILL_PRE = 0,
        PREFILL_PROC = 1,
        PREFILL_POST = 2,
        DECODE_PRE = 3,
        DECODE_PROC = 4,
        DECODE_POST = 5,
    };

    // Read N rows of "batch_size + 6 durations" (a duration of -1 means
    // "not measured at this batch size" and is simply skipped).
    void load(int n_rows) {
        vector<array<double, 7>> rows(n_rows);
        for (auto &row : rows) {
            for (double &field : row) cin >> field;
        }
        // Sort by batch size so the later binary search / interpolation works.
        sort(rows.begin(), rows.end(),
             [](const array<double, 7> &a, const array<double, 7> &b) {
                 return a[0] < b[0];
             });
        for (const auto &row : rows) {
            int b = static_cast<int>(llround(row[0]));
            for (int c = 0; c < 6; ++c) {
                double v = row[1 + c];
                if (v >= 0.0) {
                    xs_[c].push_back(b);
                    ys_[c].push_back(v);
                }
            }
        }
    }

    // Duration of task type `c` at batch size `b`. Below the smallest
    // sampled size, use the first sample; above the largest, use the last;
    // in between, linearly interpolate between the two nearest samples.
    // Memoised because this is called extremely often in the main loop.
    double interp(int c, int b) {
        auto &memo = cache_[c];
        auto it = memo.find(b);
        if (it != memo.end()) return it->second;

        const vector<int> &X = xs_[c];
        const vector<double> &Y = ys_[c];
        double r;
        if (X.empty()) {
            r = 1.0;  // Not expected: every column is guaranteed >=1 entry.
        } else if (b <= X.front()) {
            r = Y.front();
        } else if (b >= X.back()) {
            r = Y.back();
        } else {
            int i = int(lower_bound(X.begin(), X.end(), b) - X.begin());
            if (X[i] == b) {
                r = Y[i];
            } else {
                int x0 = X[i - 1], x1 = X[i];
                double y0 = Y[i - 1], y1 = Y[i];
                r = y0 + (y1 - y0) * double(b - x0) / double(x1 - x0);
            }
        }
        memo[b] = r;
        return r;
    }

private:
    array<vector<int>, 6> xs_;
    array<vector<double>, 6> ys_;
    array<unordered_map<int, double>, 6> cache_;
};

// ---------------------------------------------------------------------------
// A task currently running on the local computer E.
// ---------------------------------------------------------------------------
struct LocalTask {
    enum Kind { PPRE, PPOST, DPRE, DPOST };
    Kind kind;
    int rid = -1;        // Valid for PPRE / PPOST (single-request tasks).
    vector<int> group;   // Valid for DPRE / DPOST (across-remote groups).
};

// ---------------------------------------------------------------------------
// A task currently running on one remote computer Ck.
// ---------------------------------------------------------------------------
struct RemoteTask {
    enum Kind { PPROC, DPROC };
    Kind kind;
    int rid = -1, ls = -1, le = -1;  // Valid for PPROC (one input-stage piece).
    vector<int> group;               // Valid for DPROC.
};

// Consume and discard `n` whitespace-separated tokens from `in`. Used to
// skip the echoed task_spec/dur fields of a TDN event: we already know
// what task we issued (we recorded it ourselves), so we only need to keep
// the input stream in sync -- not re-parse the echo.
static inline void skip_tokens(istream &in, int n) {
    string dummy;
    for (int i = 0; i < n; ++i) in >> dummy;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ---------------- Startup configuration (two fixed lines) ----------------
    int K, BPT, NL;
    double S, LAT, BW;
    cin >> K >> S >> LAT >> BW >> BPT >> NL;
    (void)LAT; (void)BW; (void)BPT;  // Transfers are automatic; never needed here.

    double SLO1, SLO2, TP_UB, TP_BASE, DIST_BASE, W_TP, W_C;
    cin >> SLO1 >> SLO2 >> TP_UB >> TP_BASE >> DIST_BASE >> W_TP >> W_C;
    (void)TP_BASE;  // Only feeds the judge's own scoring, not our heuristics.

    // ---------------- Task-time table ----------------
    int n_rows;
    cin >> n_rows;
    TaskTimeTable table;
    table.load(n_rows);

    // decode_proc time for a single request: a rough proxy for how much
    // decode pressure one "active" request adds on a remote computer.
    const double DP1 = table.interp(TaskTimeTable::DECODE_PROC, 1);

    // Estimated local (edge) cost per request when D PRE/D POST are done in
    // batches of size b: two schedule costs S plus D PRE + D POST, spread
    // over the batch. Used only to size BATCH_TARGET below.
    auto local_cost_per_req = [&](int b) {
        return (2.0 * S + table.interp(TaskTimeTable::DECODE_PRE, b) +
                table.interp(TaskTimeTable::DECODE_POST, b)) / b;
    };

    // Smallest batch size whose per-request local cost is within 10% of the
    // best achievable (at batch size BMAX), capped at 256. Gives a "good
    // enough" batching target without forcing unbounded grouping.
    const int BMAX = 512;
    double ref = local_cost_per_req(BMAX);
    int BATCH_TARGET = BMAX;
    for (int b = 1; b <= BMAX; ++b) {
        if (local_cost_per_req(b) <= ref * 1.10) {
            BATCH_TARGET = b;
            break;
        }
    }
    if (BATCH_TARGET > 256) BATCH_TARGET = 256;

    // ---- General scheduling bias derived from the scoring parameters ----
    // HARD_SLO: with dist_base == 0 the waiting-time component is all-or-
    // nothing (1 only when both SLOs are exactly met), so there's no
    // "partial credit" for getting closer once we're already over.
    // ADMIT_CAP: how many requests we allow into the decode loop before we
    // start preferring throughput-mode admission, when weights favour it.
    const bool HARD_SLO = (DIST_BASE <= 0.0);
    const int ADMIT_CAP = (W_TP >= W_C) ? (1 << 30) : BATCH_TARGET;

    // ---------------- Per-request state ----------------
    // Request pipeline states (see the protocol's Request Lifecycle):
    //   0  arrived, waiting for P PRE          9  D PRE running
    //   1  P PRE running                      10  D PRE done, waiting UP
    //   2  P PRE done, waiting UP transfer     11  D PROC ready
    //   3  P PROC ready                        12  D PROC running
    //   4  P PROC running                      13  D PROC done, waiting DOWN
    //   5  P PROC done, waiting DOWN transfer  14  D POST ready
    //   6  P POST ready                        15  D POST running
    //   7  P POST running                      16  finished
    //   8  decode-ready, waiting for D PRE
    vector<int> ST;        // pipeline state
    vector<int> LIN;       // input length
    vector<int> ASG;       // assigned remote computer
    vector<int> CUR;       // next input-stage part index (piece splitting)
    vector<double> LTOK;   // timestamp the previous token was produced
    vector<double> PFULL;  // full P PROC duration for this request
    vector<double> ARRT;   // arrival timestamp
    vector<int> TOKS;      // number of output tokens produced so far

    // Grow the per-request arrays so index `rid` is valid.
    auto ensure_request = [&](int rid) {
        if ((int)ST.size() <= rid) {
            size_t n = rid + 1;
            ST.resize(n, 0);
            LIN.resize(n, 0);
            ASG.resize(n, 0);
            CUR.resize(n, 0);
            LTOK.resize(n, 0.0);
            PFULL.resize(n, 0.0);
            ARRT.resize(n, 0.0);
            TOKS.resize(n, 0);
        }
    };

    // ---------------- Ready queues per pipeline stage ----------------
    // Heaps for P PRE / P POST let us prefer shorter tasks first.
    PriorityQueueD ready_ppre, ready_ppost;
    // One P PROC queue per remote computer.
    vector<PriorityQueueD> ready_pproc(K);
    // Decode-process-ready requests, split by assigned remote computer.
    vector<set<int>> ready_dproc(K);
    // Requests ready for the local D PRE / D POST stages. Iteration order
    // doesn't matter: the protocol defines a group by its member *set*.
    set<int> ready_dpre, ready_dpost;

    // ---------------- Resource state ----------------
    // E and each Ck run at most one task at a time; presence of a task
    // record doubles as the "busy" flag (no separate bool needed).
    optional<LocalTask> task_local;
    vector<optional<RemoteTask>> task_rem(K);

    // rem_pref_ms[k] approximates outstanding input-stage work assigned to
    // remote k; dec_active[k] approximates how many requests on remote k
    // are currently in the decode loop. Both feed pick_remote() below.
    vector<double> rem_pref_ms(K, 0.0);
    vector<int> dec_active(K, 0);

    // ---------------- Live scoring telemetry ----------------
    // Running estimates of TDR/TPOT pressure, used only to steer the
    // latency-vs-throughput heuristics below -- never sent to the judge.
    double tdr_sum = 0.0; int tdr_n = 0;
    double tpot_sum = 0.0; int tpot_n = 0;
    int n_pend = 0; double sum_arr_pend = 0.0;   // not yet decode-ready
    int n_dec = 0; double sum_ltok = 0.0;        // in the decode loop
    int n_waitdown = 0;                          // decode results in flight down
    long long tokens = 0; int fin_cnt = 0, multi_cnt = 0;
    double first_arr = -1.0;

    // Pick the remote computer that currently looks least loaded, combining
    // outstanding input-stage work with a rough decode-pressure estimate.
    auto pick_remote = [&]() {
        int best = 0;
        double bv = numeric_limits<double>::max();
        for (int k = 0; k < K; ++k) {
            double v = rem_pref_ms[k] + dec_active[k] * DP1 * 20.0;
            if (v < bv) { bv = v; best = k; }
        }
        return best;
    };

    // ---------------- Main interaction loop ----------------
    string tok;
    while (cin >> tok) {
        if (tok == "END") break;  // Judge is done; nothing more to read.

        double now = stod(tok);
        int e;
        cin >> e;

        vector<int> fins;  // FIN'd requests this frame, applied after the loop.

        // Process every event in the frame before deciding anything, so
        // that everything available "now" is visible to the scheduler.
        for (int ev = 0; ev < e; ++ev) {
            string typ;
            cin >> typ;

            if (typ == "TDN") {
                string srv;
                cin >> srv;
                if (srv == "E") {
                    LocalTask tk = *task_local;
                    task_local.reset();
                    switch (tk.kind) {
                        case LocalTask::PPRE: {
                            skip_tokens(cin, 5);  // P PRE <remote> <rid> <dur>
                            ST[tk.rid] = 2;
                            break;
                        }
                        case LocalTask::PPOST: {
                            skip_tokens(cin, 5);  // P POST <remote> <rid> <dur>
                            int r = tk.rid;
                            ST[r] = 8;
                            tdr_sum += now - ARRT[r]; tdr_n += 1;
                            n_pend -= 1; sum_arr_pend -= ARRT[r];
                            n_dec += 1; LTOK[r] = now; sum_ltok += now;
                            dec_active[ASG[r]] += 1;
                            ready_dpre.insert(r);
                            break;
                        }
                        case LocalTask::DPRE: {
                            int m = (int)tk.group.size();
                            skip_tokens(cin, m + 5);  // D PRE -1 <m> <rid...> <dur>
                            for (int r : tk.group) ST[r] = 10;
                            break;
                        }
                        case LocalTask::DPOST: {
                            int m = (int)tk.group.size();
                            skip_tokens(cin, m + 5);  // D POST -1 <m> <rid...> <dur>
                            for (int r : tk.group) {
                                ST[r] = 8;
                                tokens += 1;
                                if (TOKS[r] >= 1) {
                                    tpot_sum += now - LTOK[r]; tpot_n += 1;
                                    if (TOKS[r] == 1) multi_cnt += 1;
                                }
                                TOKS[r] += 1;
                                sum_ltok += now - LTOK[r]; LTOK[r] = now;
                                ready_dpre.insert(r);
                            }
                            break;
                        }
                    }
                } else {
                    int k = stoi(srv.substr(1));
                    RemoteTask tk = *task_rem[k];
                    task_rem[k].reset();
                    if (tk.kind == RemoteTask::PPROC) {
                        skip_tokens(cin, 7);  // P PROC <ls> <le> <remote> <rid> <dur>
                        int r = tk.rid, ls = tk.ls, le = tk.le;
                        rem_pref_ms[k] -= (le - ls) / double(NL) * PFULL[r];
                        CUR[r] = le;
                        if (le < NL) {
                            // More input-stage pieces remain for this request.
                            ST[r] = 3;
                            ready_pproc[k].push({PFULL[r], r});
                        } else {
                            ST[r] = 5;  // Waiting for the input-stage downlink.
                        }
                    } else {  // DPROC
                        int m = (int)tk.group.size();
                        skip_tokens(cin, m + 5);  // D PROC <remote> <m> <rid...> <dur>
                        for (int r : tk.group) ST[r] = 13;
                        n_waitdown += m;
                    }
                }
            } else if (typ == "XDN") {
                string updown; cin >> updown;
                int k; cin >> k;
                long long size; cin >> size;  // Byte count; not needed here.
                (void)size;
                string stage; cin >> stage;   // "PRE" or "DEC"
                int m; cin >> m;
                vector<int> rids(m);
                for (int i = 0; i < m; ++i) cin >> rids[i];

                if (stage == "PRE") {
                    int r = rids[0];  // Input-stage transfers always carry one request.
                    if (updown == "UP") {
                        ST[r] = 3;
                        ready_pproc[k].push({PFULL[r], r});
                    } else {
                        ST[r] = 6;
                        ready_ppost.push({table.interp(TaskTimeTable::PREFILL_POST, LIN[r]), r});
                    }
                } else {  // "DEC"
                    if (updown == "UP") {
                        for (int r : rids) { ST[r] = 11; ready_dproc[k].insert(r); }
                    } else {
                        for (int r : rids) { ST[r] = 14; ready_dpost.insert(r); }
                        n_waitdown -= m;
                    }
                }
            } else if (typ == "ARR") {
                int rid, lin;
                cin >> rid >> lin;
                ensure_request(rid);
                ST[rid] = 0; LIN[rid] = lin; CUR[rid] = 0; TOKS[rid] = 0;
                ARRT[rid] = now;
                PFULL[rid] = table.interp(TaskTimeTable::PREFILL_PROC, lin);
                if (first_arr < 0.0) first_arr = now;
                n_pend += 1; sum_arr_pend += now;
                ready_ppre.push({PFULL[rid], rid});
            } else {  // "FIN"
                int rid;
                cin >> rid;
                fins.push_back(rid);
            }
        }

        // Apply FIN events after the whole frame has been read, matching
        // the guarantee that FIN always accompanies its final D POST's TDN.
        for (int rid : fins) {
            ST[rid] = 16;
            ready_dpre.erase(rid);
            dec_active[ASG[rid]] -= 1;
            n_dec -= 1;
            sum_ltok -= LTOK[rid];
            fin_cnt += 1;
        }

        // -------------------- Estimate current scoring pressure --------------------
        // Not needed for correctness -- only used to bias task selection.
        // Projected means include waiting time already accrued by requests
        // still in flight, so pressure builds before it's actually realised
        // in a finished measurement.
        double den = tdr_n + n_pend;
        double m_tdr = den > 0 ? (tdr_sum + n_pend * now - sum_arr_pend) / den : 0.0;

        double m_tpot;
        if (fin_cnt >= 8 && multi_cnt == 0) {
            // Every finished request so far produced exactly one token, so
            // TPOT is structurally zero: it must not steer scheduling yet.
            m_tpot = 0.0;
        } else {
            double den2 = tpot_n + n_dec;
            m_tpot = den2 > 0 ? (tpot_sum + n_dec * now - sum_ltok) / den2 : 0.0;
        }

        double r_tdr = m_tdr / SLO1;
        double r_tpot = m_tpot / SLO2;
        double ex1 = max(0.0, r_tdr - 1.0);
        double ex2 = max(0.0, r_tpot - 1.0);
        double dist = sqrt(ex1 * ex1 + ex2 * ex2);

        // Can the latency component still earn points?
        bool lat_live;
        if (W_C <= 0.0) {
            lat_live = false;
        } else if (HARD_SLO) {
            lat_live = (dist <= 0.0);
        } else {
            lat_live = (dist < DIST_BASE * 1.5);
        }

        bool thr_live = (W_TP > 0.0);
        if (thr_live && first_arr >= 0.0 && now > first_arr) {
            thr_live = (double(tokens) / (now - first_arr)) < TP_UB;
        }

        // Throughput mode: latency is either worthless or has slack to spare.
        bool TP_MODE = thr_live && (!lat_live || W_TP >= 0.7);

        bool prefer_pref;  // true => favour input-stage (prefill) work
        if (!lat_live) {
            prefer_pref = true;
        } else if (!thr_live) {
            prefer_pref = (r_tdr >= r_tpot);
        } else if (W_TP >= W_C) {
            prefer_pref = (r_tdr >= r_tpot) || (n_dec < ADMIT_CAP);
        } else {
            prefer_pref = (r_tdr >= r_tpot);
        }

        // Merging D POSTs saves one local task overhead, but waiting too
        // long can widen token gaps -- only do it in throughput mode, and
        // only when waiting would at least double the eventual group; this
        // never idles the edge, since D POST stays available in the
        // fallback order below.
        bool defer_dpost = TP_MODE && n_waitdown > 0 &&
                            n_waitdown >= (int)ready_dpost.size();

        // -------------------- Choose this frame's assignments --------------------
        vector<string> res;  // One line per assignment; sent together at the end.

        // ---- Local computer E: P PRE/P POST, D PRE, D POST all compete here ----
        if (!task_local.has_value()) {
            // 'P' = input-stage work (P POST if any, else P PRE),
            // 'D' = D PRE, 'X' = D POST. Try them in a priority order that
            // depends on whether we're currently favouring latency or
            // throughput, and whether we're deliberately deferring D POST.
            vector<char> order;
            if (prefer_pref) {
                order = defer_dpost ? vector<char>{'P', 'D', 'X'}
                                     : vector<char>{'X', 'P', 'D'};
            } else {
                order = defer_dpost ? vector<char>{'D', 'P', 'X'}
                                     : vector<char>{'X', 'D', 'P'};
            }

            for (char kind : order) {
                if (kind == 'X') {
                    if (!ready_dpost.empty()) {
                        vector<int> g(ready_dpost.begin(), ready_dpost.end());
                        ready_dpost.clear();
                        for (int r : g) ST[r] = 15;
                        task_local = LocalTask{LocalTask::DPOST, -1, g};
                        // D POST produces one token for every member of g.
                        string line = "E D POST -1 " + to_string(g.size());
                        for (int r : g) line += " " + to_string(r);
                        res.push_back(move(line));
                        break;
                    }
                } else if (kind == 'P') {
                    if (!ready_ppost.empty()) {
                        int r = ready_ppost.top().second; ready_ppost.pop();
                        ST[r] = 7;
                        task_local = LocalTask{LocalTask::PPOST, r, {}};
                        // P POST is legal only after the input-stage downlink XDN.
                        res.push_back("E P POST " + to_string(ASG[r]) + " " + to_string(r));
                        break;
                    }
                    if (!ready_ppre.empty()) {
                        int r = ready_ppre.top().second; ready_ppre.pop();
                        int k = pick_remote();
                        ASG[r] = k; ST[r] = 1;
                        rem_pref_ms[k] += PFULL[r];
                        task_local = LocalTask{LocalTask::PPRE, r, {}};
                        // P PRE fixes ASG[r] permanently: all later work for r uses Ck.
                        res.push_back("E P PRE " + to_string(k) + " " + to_string(r));
                        break;
                    }
                } else {  // 'D'
                    if (!ready_dpre.empty()) {
                        vector<int> g(ready_dpre.begin(), ready_dpre.end());
                        ready_dpre.clear();
                        for (int r : g) ST[r] = 9;
                        task_local = LocalTask{LocalTask::DPRE, -1, g};
                        // D PRE can batch requests even across different remotes;
                        // the resulting uplinks are split per remote automatically.
                        string line = "E D PRE -1 " + to_string(g.size());
                        for (int r : g) line += " " + to_string(r);
                        res.push_back(move(line));
                        break;
                    }
                }
            }
        }

        // ---- Remote computers: try to start one legal task on each free one ----
        for (int k = 0; k < K; ++k) {
            if (task_rem[k].has_value()) continue;

            set<int> &dq = ready_dproc[k];
            PriorityQueueD &pq = ready_pproc[k];
            bool use_pref;
            if (!dq.empty() && !pq.empty()) {
                use_pref = prefer_pref;
            } else if (!dq.empty()) {
                use_pref = false;
            } else if (!pq.empty()) {
                use_pref = true;
            } else {
                continue;  // Nothing ready on this remote right now.
            }

            if (use_pref) {
                int r = pq.top().second; pq.pop();
                ST[r] = 4;
                int ls = CUR[r];
                task_rem[k] = RemoteTask{RemoteTask::PPROC, r, ls, NL, {}};
                // A first solution uses one full piece [0, num_layers); to
                // experiment with splitting, this is the place to change.
                res.push_back("C" + to_string(k) + " P PROC " + to_string(ls) + " " +
                               to_string(NL) + " " + to_string(k) + " " + to_string(r));
            } else {
                vector<int> g(dq.begin(), dq.end());
                dq.clear();
                for (int r : g) ST[r] = 12;
                task_rem[k] = RemoteTask{RemoteTask::DPROC, -1, -1, -1, g};
                // All members of a D PROC group must share this remote computer.
                string line = "C" + to_string(k) + " D PROC " + to_string(k) + " " +
                              to_string(g.size());
                for (int r : g) line += " " + to_string(r);
                res.push_back(move(line));
            }
        }

        // -------------------- Send the response and flush --------------------
        // Flush is mandatory: the interactive judge will otherwise wait forever.
        if (!res.empty()) {
            cout << res.size() << '\n';
            for (const string &line : res) cout << line << '\n';
        } else {
            cout << 0 << '\n';
        }
        cout.flush();
    }

    return 0;
}
