"""Write environment.yml with the exact package versions installed in the current environment.

    conda activate resnet
    python pin_env.py

Run this once on the machine that produced the results, then commit environment.yml.
"""
import platform
from importlib.metadata import version

torch_v = version("torch")
cuda_tag = torch_v.split("+")[1] if "+" in torch_v else "cu126"
pins = {p: version(p) for p in ("torch", "torchvision", "numpy", "matplotlib")}
py = ".".join(platform.python_version_tuple()[:2])

yml = f"""# Exact versions used to produce the results in this repository (written by pin_env.py).
#   conda env create -f environment.yml
#   conda activate resnet
# PyTorch comes from the official wheel index for {cuda_tag}; the NVIDIA driver must support that CUDA version.
name: resnet
channels:
  - conda-forge
dependencies:
  - python={py}
  - pip
  - pip:
      - --extra-index-url https://download.pytorch.org/whl/{cuda_tag}
      - torch=={pins['torch']}
      - torchvision=={pins['torchvision']}
      - numpy=={pins['numpy']}
      - matplotlib=={pins['matplotlib']}
"""
open("environment.yml", "w").write(yml)
print(yml)
