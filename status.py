"""Show which experiments in reproduce.sh are finished, partly done, or not started.

    python status.py                # checks runs/
    python status.py runs_fresh     # checks another folder
"""
import csv, json, os, sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
PLAN = [
    ("Core claim (3 seeds)", [f"{a}{d}_A_s{s}" for s in (0, 1, 2) for d in (20, 56) for a in ("plain", "resnet")]),
    ("Other depths", ["plain32_A_s0", "resnet32_A_s0", "plain44_A_s0", "resnet44_A_s0", "resnet110_A_s0"]),
    ("Gap investigation", ["resnet44_A_s1", "resnet44_A_s2", "resnet56_A_s0_fp32", "resnet56_B_s0", "resnet56_B_s1", "resnet56_B_s2"]),
]

def status(name):
    d = os.path.join(RUNS, name)
    if os.path.exists(os.path.join(d, "result.json")):
        r = json.load(open(os.path.join(d, "result.json")))
        w = "" if os.path.exists(os.path.join(d, "final.pt")) else "  (final.pt missing: Fig. 7 needs it)"
        return f"done      test error {r['final_test_err']:.2f}%{w}"
    if os.path.exists(os.path.join(d, "log.csv")):
        it = 0
        try:
            rows = list(csv.reader(open(os.path.join(d, "log.csv"), errors="ignore")))[1:]
            it = int(float(rows[-1][0])) if rows else 0
        except (ValueError, IndexError):
            return "DAMAGED   log unreadable: delete this run's folder and rerun"
        ck = "checkpoint ok" if os.path.exists(os.path.join(d, "last.pt")) else "no checkpoint"
        return f"partial   iteration {it:,} of 64,000 ({ck}); rerun reproduce.sh to resume"
    return "not started"

for title, names in PLAN:
    print(f"\n{title}")
    for n in names:
        print(f"  {n:22s} {status(n)}")
