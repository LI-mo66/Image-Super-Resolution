# LFMN

LFMN is a lightweight image super-resolution project. The LFMN implementation is distributed as a compiled CPython extension in `model/lfmn.cpython-39-x86_64-linux-gnu.so`. This repository includes pretrained checkpoints for three upscaling factors and a clean evaluation script for standard benchmark datasets.

## Contents

- `model/lfmn.cpython-39-x86_64-linux-gnu.so`: compiled LFMN network extension for CPython 3.9 on Linux x86_64.
- `model/common.py`: shared model layers.
- `model/scale2_model_996.pt`: pretrained x2 checkpoint.
- `model/scale3_model_969.pt`: pretrained x3 checkpoint.
- `model/scale4_model_939.pt`: pretrained x4 checkpoint.
- `demo.sh`: evaluation script for x2, x3, and x4.
- `main.py`: main training/evaluation entry point.
- `data/`: dataset loaders.
- `trainer.py`, `utility.py`: evaluation, metrics, logging, and runtime helpers.

## Requirements

The project is written in Python and uses PyTorch.

Install the main dependencies in your environment:

```bash
pip install -r requirements.txt
```

Use the PyTorch installation command that matches your CUDA version if you need GPU support.

The compiled model extension is version-specific. Use CPython 3.9 on Linux x86_64, or rebuild the extension for a different Python/platform combination.

## Dataset Layout

By default, `demo.sh` expects benchmark datasets under:

```text
/home/huanzhujie/datasets
```

The benchmark directory should follow this structure:

```text
datasets/
└── benchmark/
    ├── Set5/
    │   ├── HR/
    │   │   ├── baby.png
    │   │   └── ...
    │   └── LR_bicubic/
    │       ├── X2/
    │       │   ├── babyx2.png
    │       │   └── ...
    │       ├── X3/
    │       │   ├── babyx3.png
    │       │   └── ...
    │       └── X4/
    │           ├── babyx4.png
    │           └── ...
    ├── Set14/
    ├── B100/
    └── Urban100/
```

The same layout is used for each benchmark dataset.

## Quick Start

Run evaluation with the default settings:

```bash
cd /home/enovo/code
./demo.sh
```

This tests x2, x3, and x4 using:

- `model/scale2_model_996.pt`
- `model/scale3_model_969.pt`
- `model/scale4_model_939.pt`

Self-ensemble is enabled by default.

## Common Evaluation Commands

Use a different dataset root:

```bash
DIR_DATA=/path/to/datasets ./demo.sh
```

Test only one scale:

```bash
SCALES=4 ./demo.sh
```

Test selected scales:

```bash
SCALES="2 4" ./demo.sh
```

Use a different benchmark set:

```bash
DATA_TEST=Set5 ./demo.sh
```

Disable self-ensemble:

```bash
SELF_ENSEMBLE=0 ./demo.sh
```

Run on CPU:

```bash
./demo.sh --cpu --n_threads 0
```

Save super-resolved images:

```bash
./demo.sh --save_results
```

## Direct `main.py` Usage

You can also run a checkpoint directly:

```bash
python main.py \
    --dir_data /path/to/datasets \
    --model LFMN \
    --data_test Set5+Set14+B100+Urban100 \
    --scale 4 \
    --pre_train model/scale4_model_939.pt \
    --test_only \
    --self_ensemble \
    --save test/lfmn_benchmark/x4
```

## Outputs

By default, logs are written under:

```text
../experiment/all_runs/test/lfmn_benchmark/
```

For example, the x4 run is saved to:

```text
../experiment/all_runs/test/lfmn_benchmark/x4/
```

If `--save_results` is enabled, reconstructed images are also saved in the corresponding result directory.

## Notes

- The repository is currently organized around LFMN only.
- `demo.sh` is evaluation-only and does not start training.
- Benchmark metrics are computed by the existing project evaluation pipeline in `trainer.py` and `utility.py`.
