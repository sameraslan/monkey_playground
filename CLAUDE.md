# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for CCN 2024 paper: "Recurrent models optimized for face recognition exhibit representational dynamics resembling the primate brain." Explores biologically-inspired recurrent neural networks (BLT architecture) for face recognition, comparing learned representations against primate neural data.

**Reference papers** are in `papers/` (CCN 2024 paper, Papale TVSD Neuron 2025).

## Training

```bash
# Single GPU
python main.py --model blt_b --dataset vggface2 --epochs 100 --batch_size 64

# Distributed (multi-GPU)
torchrun --nproc_per_node=N main.py --model blt_b --dataset vggface2

# Key flags: --optimizer {adam,sgd}, --lr, --weight_decay, --label_smoothing,
#   --times (recurrent steps), --pooling_function {max,avg,blur},
#   --objective {classification,contrastive}, --wandb (enable W&B logging)
```

There is no requirements.txt. Key dependencies: `torch`, `torchvision`, `numpy`, `scipy`, `wandb`, `tqdm`, `antialiased_cnns`, `rsatoolbox`, `repsim`, `scikit-learn`, `matplotlib`, `pandas`.

No automated test suite exists.

## Architecture

### Training Pipeline

- **main.py** — Entry point. Parses args, builds model/dataset/optimizer, runs training loop. Contains `LabelSmoothLoss` and `SetCriterion` (multi-head loss with decay weighting for recurrent readouts).
- **engine.py** — `train_one_epoch()` and `evaluate()` loops with distributed reduction and top-1/top-5 accuracy.
- **utils.py** — Distributed training helpers (`init_distributed_mode`, `reduce_dict`), `MetricLogger`, checkpoint save/load.

### Models (`models/`)

- **blt.py** — Core BLT architecture. Topology defined by a **connection matrix** specifying feedforward, feedback, and lateral connections. Supports configurable recurrent time steps and pooling modes. This is the primary model under study.
- **cornet.py** — CORnet family (Z, S, R, RT) with V1→V2→V4→IT hierarchy. Used as baselines.
- **ResNet.py** — ResNet-50 bottleneck implementation.
- **build_model.py** — Factory: `build_model(model_name, ...)` returns the appropriate architecture. Over 30 named BLT variants (e.g., `blt_b`, `blt_bl`, `blt_blt`, `blt_b_top2linear`).
- **activations.py** — Hook-based intermediate feature extraction for analysis.

### Datasets (`datasets/`)

- **vggface2.py** — VGGFace2 face identity classification. Uses face-specific normalization `[0.6068, 0.4517, 0.3800]`.
- **datasets.py** — ImageNet loader with standard augmentations. Supports subset selection via `num_cats`.
- **TVSD** (Temporal Visual Stimuli Dataset) — External dataset at https://gin.g-node.org/paolo_papale/TVSD, described in Papale et al. (Neuron 2025, see `papers/`). Used for comparing model representations against primate neural recordings.

### Analysis & Visualization

- **analyze_representations.py** — RDM computation, Angular CKA similarity, PCA/manifold analysis. Extracts layer activations and compares against neural data using `rsatoolbox`.
- **tikz_visualizer.py** — Generates TikZ diagrams of network architectures.
- **notebooks/** — Jupyter notebooks for tuning dynamics analysis, model visualization, and face patch analysis.

## Tool Usage Rules

- **Never chain Bash commands** with `&&`, `|`, or `;`. Run each command as a separate tool call, or use the dedicated tools instead (Read instead of cat, Glob instead of find, Grep instead of grep). Chained commands bypass the user's permission allow-list.
- Always run commands from the project root. Do not use cd in compound commands. Use absolute paths instead.

## Key Design Patterns

- BLT model topology is defined by connection matrices — changing the matrix changes which layers connect and how (feedforward, feedback, lateral). This enables systematic architecture search.
- `SetCriterion` applies loss at each recurrent time step with exponential decay weighting, encouraging the network to solve the task across time.
- Distributed training is built-in via PyTorch DDP; single-GPU works without any flags.
