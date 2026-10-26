# resnet-repro

Reproduction of one claim from He, Zhang, Ren and Sun (2016), *Deep Residual Learning for Image Recognition* (CVPR), done for the Wits MSc reproduction assignment. The write-up is `paper/main.pdf`.

## Claim

On CIFAR-10, plain networks get worse as they get deeper (both training and test error go up, which the paper calls the degradation problem). Residual networks with parameter-free identity shortcuts, at exactly the same depth, width and parameter count, get better as they get deeper.

Source: Section 4.2 of the paper, Figure 6 (left: plain nets with 20 to 56 layers; middle: ResNets with 20 to 110 layers) and Table 6 (ResNet test error of 8.75% at 20 layers, 7.51% at 32, 7.17% at 44, 6.97% at 56 and 6.43% at 110).

## Route and authorship

I reimplemented the method in PyTorch from the paper's text. I did not use the authors' original Caffe code.

The code was written with the help of an AI assistant. I chose the claim and the extra experiments (more seeds, the FP32 control, option B shortcuts), ran every training run on my own laptop, checked the architecture, schedule and augmentation against the paper, fixed problems that came up while training (for example a run whose log was corrupted by a power cut), and interpreted the results.

## Reproducing everything

```bash
git clone https://github.com/Royal93/resnet-repro.git
cd resnet-repro
conda env create -f environment.yml
conda activate resnet
bash reproduce.sh
```

`reproduce.sh` is the single command behind every number, table and figure in the write-up.

- On a clean checkout it finds the committed runs in `runs/` (logs, results and final weights), skips the training, and rebuilds everything in `results/` in a few minutes. If `pdflatex` is installed it also rebuilds `paper/main.pdf`.
- To retrain every model from scratch into separate folders (about 13 hours in total on one RTX 3050 laptop GPU with 4 GB; the times of each run are in `runs/*/result.json`):

  ```bash
  RUNS=runs_fresh OUT=results_fresh bash reproduce.sh
  ```

- Training can be stopped and restarted at any point. Finished runs are skipped and an interrupted run carries on from its last checkpoint. `python status.py` shows what has finished.
- The only download is CIFAR-10 (about 170 MB). `torchvision` fetches it into `data/` the first time it is needed, including for `layer_responses.py` (Fig. 7).
- Training needs an NVIDIA GPU. The analysis step also runs on a CPU.

## Results

The full table, with every seed, is in `results/summary.md`. CIFAR-10 test error (%), mean ± std over 3 seeds where there are 3:

| Depth | Plain (mine) | ResNet, option A (mine) | ResNet (paper, Table 6) |
|---|---|---|---|
| 20 | 9.56 ± 0.20 | 8.36 ± 0.27 | 8.75 |
| 32 | 9.89 (1 run) | 7.78 (1 run) | 7.51 |
| 44 | 10.94 (1 run) | 7.41 ± 0.30 | 7.17 |
| 56 | 12.79 ± 0.41 | 7.93 ± 0.23 | 6.97 |
| 110 | not run | 6.50 (1 run, with warm-up) | 6.43 (best of 5; mean 6.61 ± 0.16) |

What reproduced: plain networks degrade with depth (3.2 points worse from 20 to 56 layers), residual networks improve with depth overall from 20 to 110 layers, and ResNet-110 matches the paper.

What did not: with option A shortcuts, ResNet-56 is 0.96 points worse than the paper and also worse than ResNet-44 (3 seeds each).

Extra runs that the paper does not report: ResNet-56 trained in full FP32 gets 7.95%, the same as with mixed precision. ResNet-56 with option B (projection) shortcuts gets 7.06 ± 0.33% over 3 seeds, which matches the paper's 6.97%.

Figures: `results/fig6_side_by_side.png` (training curves in the style of the paper's Fig. 6), `results/fig_depth.png` (test error against depth) and `results/fig7_layer_std.png` (layer response strength, as in Fig. 7).

## Files

| File | What it does |
|---|---|
| `models.py` | Plain and residual CIFAR networks: 6n+2 layers, 16/32/64 filters, option A (zero-padding) or B (projection) shortcuts |
| `train.py` | Trains one model with the paper's recipe, logs the curves, checkpoints every 500 iterations and resumes automatically |
| `reproduce.sh` | Runs every training job, then the analysis, then builds the PDF |
| `analyze.py` | Results tables (Markdown and LaTeX), Fig. 6 curves, error-against-depth plot, and the LaTeX number macros |
| `layer_responses.py` | Fig. 7: standard deviation of every 3×3 layer's response, computed from the saved models |
| `status.py` | Lists finished, partial and missing runs |
| `time_check.sh` | Short timing run to estimate training time on another machine |
| `pin_env.py` | Writes `environment.yml` from the versions installed in the active environment |
| `runs/<name>/` | One folder per run: `config.json` (settings, library versions, GPU), `log.csv` (curves), `result.json` (final numbers), `final.pt` (weights) |
| `results/` | Generated tables and figures, plus `numbers.tex` and `numbers_fig7.tex`, which the write-up reads its numbers from |
| `paper/` | LaTeX source of the write-up (IEEE conference template) |

## Setup notes

1. Check the NVIDIA driver with `nvidia-smi` (the CUDA version is shown top right). It must support CUDA 12.6 or later for the pinned PyTorch wheel.
2. `conda env create -f environment.yml`, then `conda activate resnet`.
3. Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.
4. `python models.py` prints the parameter counts (0.27M, 0.46M, 0.66M, 0.85M, 1.73M), which match Table 6 and are the same for plain and residual nets.
5. Optional: `bash time_check.sh` estimates how long a full run takes on your machine.

## Hardware and settings

- One NVIDIA RTX 3050 Laptop GPU (4 GB), Ubuntu, PyTorch 2.14.1 with CUDA 12.6. The paper used Caffe on two GPUs.
- Same as the paper: 6n+2 architecture, option A shortcuts, SGD with momentum 0.9, weight decay 0.0001, batch 128, learning rate 0.1 divided by 10 at 32k and 48k iterations, 64k iterations in total, He initialisation, batch normalisation, no dropout, 4-pixel padding with random 32×32 crops and horizontal flips, per-pixel mean subtraction, a single view of the 10,000 test images at test time, training on all 50,000 training images, and for ResNet-110 a warm-up at learning rate 0.01 until training error drops below 80%.
- Different from the paper: PyTorch instead of Caffe, mixed precision (FP16) by default (one FP32 run checks the effect), non-deterministic cuDNN kernels for speed, the final rather than the best test error is reported, one GPU instead of two, and plain-net reference values are read off the paper's Fig. 6 because Table 6 only lists ResNets.

## Data ethics

CIFAR-10 has 60,000 32×32 images in ten object classes and no person class. The images come from the 80 Million Tiny Images collection, which was scraped from web searches without the consent of the photographers or of anyone shown in the pictures. Its creators withdrew that collection in 2020 after offensive labels were found in it. CIFAR-10 itself was relabelled by hand and is still publicly available, but its web page gives no formal licence. The write-up discusses this further.

## Reference

He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.
