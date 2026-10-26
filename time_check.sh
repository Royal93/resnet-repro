#!/usr/bin/env bash
# Times a short run of each model so you can estimate how long the full runs will take.
# A full run is 64,000 iterations, so: full minutes ~= (minutes for 400 iterations) x 160.
cd "$(dirname "$0")"
for A in plain resnet; do for D in 20 56; do
  python train.py --arch $A --depth $D --iters 400 --eval-every 200 --tag timing --out-dir runs_timing
done; done
python - << 'PY'
import json, glob
for p in sorted(glob.glob("runs_timing/*/result.json")):
    r = json.load(open(p))
    print(f"{r['name']:28s} 400 iters: {r['minutes']:.2f} min  -> full 64k run approx {r['minutes']*160/60:.1f} hours")
PY
