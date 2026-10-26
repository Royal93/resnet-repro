"""
Train a plain or residual network on CIFAR-10 with the recipe from He et al. (2016), Sec. 4.2.

Paper recipe (defaults below):
    SGD, momentum 0.9, weight decay 1e-4, batch 128
    lr 0.1, divided by 10 at 32k and 48k iterations, stop at 64k
    augmentation: pad 4 pixels, random 32x32 crop, random horizontal flip
    per-pixel mean subtraction, BN, He init, no dropout
    test: single view of the original 32x32 image

Examples:
    python train.py --arch resnet --depth 20 --seed 0
    python train.py --arch plain  --depth 56 --seed 1
    python train.py --arch resnet --depth 110 --warmup      # paper's warm-up for ResNet-110
    python train.py --arch resnet --depth 20 --iters 400 --eval-every 200 --tag smoke   # quick test

Outputs go to runs/<arch><depth>_<shortcut>_s<seed>[_<tag>]/:
    log.csv        iteration, lr, train error, train loss, test error, test loss, elapsed time
    config.json    every setting plus library versions and GPU name
    last.pt        checkpoint (used to resume automatically if the run is interrupted)
    final.pt       final model weights
    result.json    final test error and parameter count
"""
import argparse
import csv
import json
import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models import build_model, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["plain", "resnet"], required=True)
    p.add_argument("--depth", type=int, required=True, help="6n+2: 20, 32, 44, 56, 110")
    p.add_argument("--shortcut", choices=["A", "B"], default="A")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=64000)
    p.add_argument("--milestones", type=int, nargs="+", default=[32000, 48000])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--warmup", action="store_true",
                   help="use lr 0.01 until training error < 80%% (paper, ResNet-110), then lr")
    p.add_argument("--val-split", action="store_true",
                   help="train on 45k, evaluate on held-out 5k (the paper's split for choosing the schedule)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--deterministic", action="store_true",
                   help="bit-for-bit reproducible but slower (disables cudnn autotuning)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--tag", default="")
    return p.parse_args()


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def load_cifar(data_dir, device, val_split, seed):
    """Load CIFAR-10 once onto the GPU as float tensors (fast: no CPU data loader needed)."""
    tr = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    te = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).float().div(255)
    ytr = torch.tensor(tr.targets)
    xte = torch.tensor(te.data).permute(0, 3, 1, 2).float().div(255)
    yte = torch.tensor(te.targets)

    if val_split:  # 45k / 5k split, fixed by seed 0 so all runs share the same split
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(xtr), generator=g)
        xte, yte = xtr[perm[45000:]], ytr[perm[45000:]]
        xtr, ytr = xtr[perm[:45000]], ytr[perm[:45000]]

    mean = xtr.mean(dim=0, keepdim=True)  # per-pixel mean image (paper: "per-pixel mean subtracted")
    xtr, xte = xtr - mean, xte - mean
    return xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)


def augment(x):
    """Pad 4 pixels with zeros, random 32x32 crop, random horizontal flip (all on the GPU)."""
    b, c, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (b,), device=x.device)
    j = torch.randint(0, 9, (b,), device=x.device)
    ar = torch.arange(h, device=x.device)
    rows = (i[:, None] + ar)[:, None, :, None]
    cols = (j[:, None] + ar)[:, None, None, :]
    out = padded[torch.arange(b, device=x.device)[:, None, None, None],
                 torch.arange(c, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(b, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


@torch.no_grad()
def evaluate(model, x, y, amp, bs=1000):
    model.eval()
    wrong, loss = 0, 0.0
    for k in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x[k:k + bs])
        loss += F.cross_entropy(out.float(), y[k:k + bs], reduction="sum").item()
        wrong += (out.argmax(1) != y[k:k + bs]).sum().item()
    model.train()
    return 100.0 * wrong / len(x), loss / len(x)


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Check `nvidia-smi` and your PyTorch install (see README).")
    device = "cuda"
    amp = not args.no_amp
    set_seed(args.seed, args.deterministic)

    name = f"{args.arch}{args.depth}_{args.shortcut}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "result.json")):
        print(f"{name}: already finished, skipping (delete {out} to rerun).")
        return

    xtr, ytr, xte, yte = load_cifar(args.data_dir, device, args.val_split, args.seed)
    model = build_model(args.arch, args.depth, args.shortcut).to(device).to(memory_format=torch.channels_last)
    n_params = count_params(model)

    # Weight decay on all parameters, as in the original Caffe setup.
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_it, elapsed, warm = 0, 0.0, args.warmup
    ckpt_path = os.path.join(out, "last.pt")
    log_path = os.path.join(out, "log.csv")
    if os.path.exists(ckpt_path):  # resume after an interruption
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:  # e.g. power cut while the file was being written
            raise SystemExit(f"{name}: checkpoint {ckpt_path} is unreadable ({e}).\n"
                             f"Try the backup: mv {ckpt_path}.bak {ckpt_path}  (or delete {out} to restart this run).")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_it, elapsed, warm = ck["it"], ck["elapsed"], ck["warm"]
        torch.set_rng_state(ck["rng_cpu"]); torch.cuda.set_rng_state(ck["rng_cuda"])
        # Drop log rows written after the checkpoint (they will be recomputed), so curves have no duplicates.
        with open(log_path) as f:
            rows = list(csv.reader(f))
        kept = [rows[0]] + [r for r in rows[1:] if int(float(r[0])) <= start_it]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerows(kept)
        print(f"{name}: resuming from iteration {start_it}")
    else:
        cfg = vars(args) | {
            "params": n_params, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
        }
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "lr", "train_err", "train_loss", "test_err", "test_loss", "elapsed_s"])

    def lr_at(it):
        if warm:
            return 0.01
        return args.lr * (0.1 ** sum(it >= m for m in args.milestones))

    print(f"{name}: {n_params/1e6:.3f}M params, {args.iters} iterations, AMP={amp}")
    n = len(xtr)
    perm, pos = torch.randperm(n, device=device), 0
    win_wrong = torch.zeros((), device=device)  # accumulated on the GPU to avoid slow syncs
    win_loss = torch.zeros((), device=device)
    win_seen = 0
    recent = []  # last 20 minibatch error rates, used only for the warm-up check
    t0 = time.time()

    for it in range(start_it, args.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        if pos + args.batch > n:  # new epoch: reshuffle
            perm, pos = torch.randperm(n, device=device), 0
        idx = perm[pos:pos + args.batch]; pos += args.batch
        xb = augment(xtr[idx]).contiguous(memory_format=torch.channels_last)
        yb = ytr[idx]

        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            wrong = (logits.argmax(1) != yb).sum()
            win_wrong += wrong; win_loss += loss.detach() * len(yb); win_seen += len(yb)

        # Paper's warm-up for ResNet-110: lr 0.01 until training error < 80% (about 400 iterations),
        # then back to 0.1. Checked on the average of the last 20 minibatches.
        if warm:
            recent = (recent + [100.0 * wrong.item() / len(yb)])[-20:]
            if len(recent) == 20 and sum(recent) / 20 < 80.0:
                warm = False
                print(f"  warm-up finished at iteration {it + 1}")

        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
            test_err, test_loss = evaluate(model, xte, yte, amp)
            el = elapsed + time.time() - t0
            row = [it + 1, lr_at(it), round(100.0 * win_wrong.item() / win_seen, 3),
                   round(win_loss.item() / win_seen, 5), test_err, round(test_loss, 5), round(el, 1)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"  it {it+1:6d}  lr {row[1]:.4f}  train err {row[2]:6.2f}%  test err {test_err:6.2f}%  "
                  f"[{el/60:.1f} min]", flush=True)
            win_wrong.zero_(); win_loss.zero_(); win_seen = 0
            # Write to a temporary file, then rename: a power cut can never leave a half-written checkpoint.
            tmp = ckpt_path + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "it": it + 1, "elapsed": el, "warm": warm,
                        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, tmp)
            if os.path.exists(ckpt_path):
                os.replace(ckpt_path, ckpt_path + ".bak")
            os.replace(tmp, ckpt_path)

    test_err, test_loss = evaluate(model, xte, yte, amp)
    torch.save(model.state_dict(), os.path.join(out, "final.pt"))
    result = {"name": name, "arch": args.arch, "depth": args.depth, "shortcut": args.shortcut,
              "seed": args.seed, "params": n_params, "final_test_err": test_err,
              "final_test_loss": test_loss, "minutes": round((elapsed + time.time() - t0) / 60, 1),
              "eval_set": "val5k" if args.val_split else "test10k", "iters": args.iters}
    json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=2)
    for p in (ckpt_path, ckpt_path + ".bak"):
        if os.path.exists(p):
            os.remove(p)
    print(f"{name}: final test error {test_err:.2f}%")


if __name__ == "__main__":
    main()
