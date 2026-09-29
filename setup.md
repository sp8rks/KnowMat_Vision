# Docling Figure Extraction: Setup

This guide sets up the Python environment for `extract_figure_docling.py` and `run_fig.sh`. Go to the section for your operating system and follow it from start to finish.

- [macOS](#macos)
- [Windows](#windows)
- [Linux](#linux)

---

## macOS

Run everything in Terminal. The script uses the Mac GPU (MPS).

### 1. Conda environment

Create and activate a `VLMtrain` env:

```bash
conda create -n VLMtrain python=3.11 pip -y
conda activate VLMtrain
```

Next, confirm that `python` and `pip` come from the active env:

```bash
which python pip
```

Both paths should include `.../envs/VLMtrain/bin/`. If they don't, the env has no pip of its own, and `pip install` will put packages somewhere else.

### 2. Install PyTorch and Docling

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install docling
```

- `python -m pip` makes sure the pip you run is the one for the active env.
- These `torch` wheels already support the Mac GPU (MPS), so no extra flags are needed.
- Docling also installs `pypdfium2`, which the script uses to render the figure PNGs at full resolution. You don't need to install anything else.

### 3. Download the Docling models (one time)

```bash
docling-tools models download
```

### 4. Verify

```bash
python -c "import docling, pypdfium2, torch; print('docling OK'); print('MPS:', torch.backends.mps.is_available())"
```

You should see:

```
docling OK
MPS: True
```

In `run_fig.sh`, set `DEVICE="mps"`.

### Troubleshooting

| Problem | Fix |
|---|---|
| `pip: command not found`, or `which pip` points outside the env | Run `conda install -y python=3.11 pip` inside `VLMtrain` |
| `MPS: False`, or MPS errors at runtime | Run with `--device cpu` (set `DEVICE="cpu"` in `run_fig.sh`) |

---

## Windows

Run everything in **Anaconda Prompt** (Start menu → "Anaconda Prompt" or "Miniconda Prompt"). `conda activate` may not work in a plain `cmd` or PowerShell window.

### 1. Conda environment

Create and activate a `VLMtrain` env:

```bat
conda create -n VLMtrain python=3.11 pip -y
conda activate VLMtrain
```

Next, confirm that `python` and `pip` come from the active env:

```bat
where python pip
```

The **first** path listed for each should include `...\envs\VLMtrain\`. If it doesn't, the env has no pip of its own, and `pip install` will put packages somewhere else.

### 2. Install PyTorch and Docling

Pick the PyTorch command that matches your machine.

**With an NVIDIA GPU:**

```bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install docling
```

First run `nvidia-smi`. The "CUDA Version" it shows must be 12.4 or higher for the `cu124` wheels. If it is lower, update the NVIDIA driver.

**Without an NVIDIA GPU (CPU only):**

```bat
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install docling
```

- Install `torch` **before** `docling`. If you don't, pip picks a default `torch` build, and on Windows that may be CPU-only.
- Docling also installs `pypdfium2`, which the script uses to render the figure PNGs at full resolution. You don't need to install anything else.

### 3. Download the Docling models (one time)

```bat
docling-tools models download
```

### 4. Verify

```bat
python -c "import docling, pypdfium2, torch; print('docling OK'); print('CUDA:', torch.cuda.is_available())"
```

You should see:

```
docling OK
CUDA: True
```

On a CPU-only machine, `CUDA: False` is expected.

### 5. Running `run_fig.sh` on Windows

`run_fig.sh` is a bash script. It runs in **Git Bash** or **WSL**, not in `cmd` or PowerShell. In `run_fig.sh`:

- Set `DEVICE="cuda"`. Without an NVIDIA GPU, set `DEVICE="cpu"`.
- Set `PYTHON` to the full path of the env's Python, which is the first line of `where python`. For example: `PYTHON="/c/Users/<you>/miniconda3/envs/VLMtrain/python.exe"`.

Without Git Bash or WSL, skip the script and call the extractor directly from Anaconda Prompt:

```bat
python extract_figure_docling.py paper.pdf --device cuda -o results
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `conda` is not recognised | Use Anaconda Prompt, or run `conda init` once and reopen the terminal |
| `where pip` lists another Python first | Run `conda install -y python=3.11 pip` inside `VLMtrain`, and always use `python -m pip` |
| `CUDA: False` on a machine with an NVIDIA GPU | Remove torch with `python -m pip uninstall -y torch torchvision`, then run the NVIDIA command from step 2 again |
| `OSError: [WinError 126]` or DLL load errors when importing torch | Install the latest [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Model download fails with symlink errors | Turn on Developer Mode (Settings → System → For developers), or run Anaconda Prompt as Administrator |

---

## Linux

Run everything in a terminal.

### 1. Conda environment

Create and activate a `VLMtrain` env:

```bash
conda create -n VLMtrain python=3.11 pip -y
conda activate VLMtrain
```

Next, confirm that `python` and `pip` come from the active env:

```bash
which python pip
```

Both paths should include `.../envs/VLMtrain/bin/`. If they don't, the env has no pip of its own, and `pip install` will put packages somewhere else.

### 2. Install PyTorch and Docling

Pick the PyTorch command that matches your machine.

**With an NVIDIA GPU:**

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install docling
```

First run `nvidia-smi`. The "CUDA Version" it shows must be 12.4 or higher for the `cu124` wheels. If it is lower, update the NVIDIA driver.

**Without an NVIDIA GPU (CPU only):**

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install docling
```

- Install `torch` **before** `docling`, so pip keeps the build you chose.
- The CPU-only wheels are much smaller than the default Linux `torch`, which includes the CUDA libraries.
- Docling also installs `pypdfium2`, which the script uses to render the figure PNGs at full resolution. You don't need to install anything else.

### 3. Download the Docling models (one time)

```bash
docling-tools models download
```

### 4. Verify

```bash
python -c "import docling, pypdfium2, torch; print('docling OK'); print('CUDA:', torch.cuda.is_available())"
```

You should see:

```
docling OK
CUDA: True
```

On a CPU-only machine, `CUDA: False` is expected.

In `run_fig.sh`, set `DEVICE="cuda"`. Without an NVIDIA GPU, set `DEVICE="cpu"`.

### Troubleshooting

| Problem | Fix |
|---|---|
| `pip: command not found`, or `which pip` points outside the env | Run `conda install -y python=3.11 pip` inside `VLMtrain` |
| `CUDA: False` on a machine with an NVIDIA GPU | Check that `nvidia-smi` works. Then remove torch with `python -m pip uninstall -y torch torchvision` and run the NVIDIA command from step 2 again |
| `libGL.so.1: cannot open shared object file` | Install the system OpenGL libraries: `sudo apt-get install -y libgl1 libglib2.0-0` (Debian/Ubuntu) |
| CUDA out of memory | Run with `--device cpu`, or process fewer PDFs at a time |
