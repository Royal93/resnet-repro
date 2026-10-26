#!/usr/bin/env bash
# Produces every number, table and figure in the write-up.
#
#   bash reproduce.sh
#       Uses runs/. Runs whose result.json is already committed are reused, so on a clean
#       checkout this only re-does the analysis (minutes). Missing runs are trained.
#
#   RUNS=runs_fresh OUT=results_fresh bash reproduce.sh
#       Retrains every model from scratch into a new folder, then analyses it.
#       About 13 hours on one RTX 3050 laptop GPU (4 GB). Needs internet once for CIFAR-10.
#
# Safe to stop and restart: finished runs are skipped, interrupted runs resume from a checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
RUNS=${RUNS:-runs}
OUT=${OUT:-results}
t () { python train.py --out-dir "$RUNS" "$@"; }

# 1. Core claim (Fig. 6, Table 6): plain vs ResNet, 20 and 56 layers, 3 seeds each.
for SEED in 0 1 2; do
  for D in 20 56; do
    t --arch plain  --depth $D --seed $SEED
    t --arch resnet --depth $D --seed $SEED
  done
done

# 2. Remaining depths of Fig. 6 / Table 6 (seed 0), ResNet-110 with the paper's warm-up.
for D in 32 44; do
  t --arch plain  --depth $D --seed 0
  t --arch resnet --depth $D --seed 0
done
t --arch resnet --depth 110 --seed 0 --warmup

# 3. Investigating the gap (not in the paper): extra seeds for ResNet-44, FP32, option B shortcuts (3 seeds).
t --arch resnet --depth 44 --seed 1
t --arch resnet --depth 44 --seed 2
t --arch resnet --depth 56 --seed 0 --no-amp --tag fp32
t --arch resnet --depth 56 --seed 0 --shortcut B
t --arch resnet --depth 56 --seed 1 --shortcut B
t --arch resnet --depth 56 --seed 2 --shortcut B

# 4. Tables, figures and LaTeX number macros.
python analyze.py         --runs "$RUNS" --out "$OUT"
python layer_responses.py --runs "$RUNS" --out "$OUT"

# 5. Build the write-up PDF (only if LaTeX is installed and the default output folder is used).
if [ "$OUT" = "results" ] && command -v pdflatex >/dev/null 2>&1; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null) \
     && echo "Write-up built: paper/main.pdf"
fi
echo "Done. Numbers and figures are in $OUT/"
