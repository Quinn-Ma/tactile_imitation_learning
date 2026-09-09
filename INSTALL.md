# Environment setup

Run all commands from the repository root.

The study used three Python environments, recorded in `ENVIRONMENT.json`. The requirement files here list observed versions, including the renderer libraries; they are not complete dependency lock files. A fresh installation of every dependency has not been revalidated for this repository release.

## Simulator environment

Use Python 3.12, with a working OpenGL/GLFW context:

```text
python -m pip install -r requirements-simulation.txt
python outputs/simulation/generate_dataset.py --help
```

The reference simulation used Python 3.12.14. The adapter selects GLFW for rendering and creates its temporary files inside this study's `work/` directory. Remote headless machines still need an appropriate graphics context; the renderer is not silently changed to another backend.

## Data conversion and analysis environment

The reference environment used Python 3.14.2:

```text
python -m pip install -r requirements-analysis.txt
python outputs/research/prepare_public_force_data.py --help
```

## CUDA training environment

Use a separate CUDA-enabled PyTorch environment. The executed runs used Python 3.14.2, PyTorch `2.14.0+cu130`, CUDA 13.0 and NumPy 2.5.2 on an RTX 5090 Laptop GPU. Install a PyTorch build compatible with the GPU, driver, and Python interpreter; `ENVIRONMENT.json` records the experimental build rather than asserting availability of that exact wheel from every package index.

Check that the intended interpreter can use CUDA:

```text
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python outputs/world_model/train.py --help
```

The training scripts require CUDA when `--device cuda` is used. In the full reproduction commands, replace `PATH_TO_TRAIN_PYTHON` with this environment's Python executable; simulator matrix runners use it to start the model/policy process. They do not store a private machine's interpreter path.

Continue with the [reproduction instructions](README_reproduction.md) for the fixed data downloads, model training, both control cohorts, and analysis. Raw data and trained checkpoints are generated locally. The hash check below is independent of these environments and needs only Python's standard library:

```text
python verify_package.py
```

`FILES_SHA256.json` covers the frozen 206-file arXiv code/results package. The original package README is retained as README_reproduction.md, and its entry is mapped accordingly in FILES_SHA256.json. The original index is ARXIV_FILES_SHA256.json. This installation note and the requirements files are repository additions; the arXiv ZIP remains unchanged.
