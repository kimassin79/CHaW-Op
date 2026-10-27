# A geometry-aware deep learning surrogate for computational fluid dynamics enables real-time hemodynamic risk stratification of abdominal aortic aneurysms

CHaW-Op maps a CTA-derived aneurysm mesh to pressure, velocity, wall shear stress, time-averaged wall shear stress, oscillatory shear index, endothelial-cell-activation potential, an intraluminal-thrombus low-shear map, peak wall stress and rupture potential index. A shared geometry-aware operator carries the field representation. The wall branch is directionally conditioned on predicted ILT and ECAP, and the velocity branch is restricted by a differentiable solenoidal projection. Outcome labels remain separated from field-surrogate training through a patient-disjoint biomarker stage.

## Scope

The source tree contains training, calibration and held-out evaluation components. It intentionally excludes clinical deployment interfaces and patient-level prediction commands. Raw CTA imaging is not part of this repository.

## Code and data availability

The repository contains the geometric tokenizer (`code/chaw_op/geometry/tokenization.py`), neural operator and ILT-to-wall coupling (`code/chaw_op/operator/`), projection (`code/chaw_op/physics/projection.py`), and biomarker and clinical evaluation functions (`code/chaw_op/operator/biomarker.py` and `code/chaw_op/evaluation/clinical.py`). The training entry point and experiment configurations are in `code/chaw_op/commands/train.py` and `experiments/`. These components do not constitute a complete automated reproduction pipeline: the repository does not provide the CFD or finite-element solvers used to generate target fields, a patient-level inference or clinical evaluation command, or a fully wired two-stage biomarker workflow.

No trained model weights or pretrained checkpoints are included in this repository. Training can write checkpoints under `artifacts/`; these are outputs of a local run, not provided pretrained models. The numerical values below are study targets and have not been independently reproduced from the files distributed here.

Patient CT scans, clinical records, derived cohort meshes and solver fields are not redistributed. Access to private study data is governed by the relevant data-use agreements; an application procedure or contact is not specified in this repository. Public datasets listed in `dataset_links.txt` have their own access conditions and are not substitutes for the study cohort. Reproducing study results requires access to the appropriate private data, target-generation methods, configurations and trained models; the released code alone is insufficient.

### Two-center study cohorts

1,134 patients · 2 centers · 68 rupture-related events

- Center-1: 928 patients (638 training, 100 validation and 190 internal-test cases).
- Center-2: 206 patients (prospective silent-mode cohort).

Derived cohort assets require the study data agreement.

## Environment

The pinned environment uses Python 3.9.15, PyTorch 1.13.1 and CUDA 11.6. It requires the matching binary builds of `torch-scatter` and `torch-sparse`. The supported host configuration is Linux x86_64 with an NVIDIA driver compatible with CUDA 11.6.

Conda installation:

```bash
conda env create -f environment.yml
conda activate chawop
python -m pip install --no-deps .
```

Container build:

```bash
docker build -t chawop:0.1.0 .
```

The fixed dependency set is deliberate because geometric sparse operators, CUDA extensions and the distributed runtime must use matching application binary interfaces.

## Data layout

Public dataset entry points and their verified access terms are collected in `dataset_links.txt`. The primary AAA-HemoNet layer is private. Sensitive raw scans are not redistributed. The training layer expects derived meshes and solver fields only.

```text
data/aaa_hemonet/
  manifests/cohort.csv
  meshes/<patient>.h5
  fields/<patient>.h5
```

Each mesh file contains `position`, `normal`, `curvature`, `wall_distance`, `edge_index`, `volume_cells` and `surface_cells`. Each field file contains `pressure`, `velocity`, `wss`, `tawss`, `osi`, `ecap`, `ilt`, `pws` and `rpi`. Positions use metres, pressure and stress use pascals, velocity uses metres per second, WSS uses pascals, and OSI and RPI are dimensionless.

The cohort manifest columns are:

```text
patient_id,site,region,split,mesh_path,field_path,diameter_mm,sex,age,event,event_time_days,scanner_vendor
```

Only pre-rupture baseline meshes belong in the manifest. A patient may occur in exactly one split. Center-2 remains prospective silent-mode evaluation data. Public pretraining geometries must be checked against private evaluation geometry hashes before optimization.

The finite-volume targets use pulsatile incompressible Navier–Stokes flow, rigid walls, patient-scaled inflow and three-element Windkessel outlets. Wall targets use finite-element analysis with a population hyperelastic wall model. Solver and mesh-resolution settings must remain constant across the cohort.

## Model

The default operator has width 256, six multi-scale operator layers, eight attention heads, 32 geometric-algebra channels and a three-layer ILT-to-wall-stress coupling branch. Four geodesic neighborhoods cover sac-scale, neck-scale, bifurcation-scale and near-wall interactions. The flow branch decodes volumetric mean velocity before the Leray projection. Surface WSS, TAWSS, OSI and ECAP are computed from the projected field and learned residual corrections. The wall branch cross-attends from geometry tokens to the model's own ILT and ECAP predictions.

The objective applies inverse-variance field balancing to all nine targets. The physics residual weight is zero through epoch 50, rises linearly to 0.1 at epoch 200 and remains fixed through epoch 300. The quantile term has weight 0.05. Functional split-conformal calibration reserves 15 percent of the development allocation.

## Training

The reported optimizer settings are AdamW, learning rate `3e-4`, weight decay `1e-5`, batch size four per optimization input and 300 epochs over ten fixed seeds. Precision, GPU model, GPU count, scheduler, warmup, gradient accumulation and gradient clipping were not stated in the manuscript. The release configuration fixes those operational choices explicitly: FP32, cosine decay, 50 warmup epochs, accumulation eight and clipping at 1.0.

The distributed allocation is 32 GPUs. With batch size four per rank and accumulation eight, the operational effective batch is 1,024 meshes. Each rank prefetches data through 16 workers. A full ten-seed run therefore requires 320 GPU allocations and stores the complete nine-field validation outputs for calibration and across-seed analysis. GPU type, VRAM, wall-clock time and storage consumption were not reported and must be measured on the executing cluster; no unsupported estimate is presented as a paper value.

Single-node launch:

```bash
bash tools/launch_train.sh
```

Multi-node launch:

```bash
torchrun --nnodes=4 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=<host>:29400 -m chaw_op.commands.train
```

Run each configured seed independently and retain patient-disjoint calibration outputs:

```bash
python -m chaw_op.commands.train seed=1701
python -m chaw_op.commands.train seed=2718
python -m chaw_op.commands.train seed=3141
python -m chaw_op.commands.train seed=4099
python -m chaw_op.commands.train seed=5279
python -m chaw_op.commands.train seed=6211
python -m chaw_op.commands.train seed=7103
python -m chaw_op.commands.train seed=8089
python -m chaw_op.commands.train seed=9001
python -m chaw_op.commands.train seed=10007
```

## Ablations

The coupling-free control replaces directional ILT-to-wall conditioning with a parallel wall decoder while keeping tokenization, shared latent processing and field heads fixed:

```bash
python -m chaw_op.commands.train --config-name coupling_free
```

The conservation control removes the Leray projection:

```bash
python -m chaw_op.commands.train --config-name no_projection
```

The `no_equivariance` configuration is present, but its `equivariant_tokenization` setting is not connected to the model. It therefore does not currently run the intended tokenization ablation.

## Evaluation contract

Field evaluation reports coefficient of determination, normalized mean absolute error, relative L2 error and vector cosine alignment. Across-seed results use the mean and sample standard deviation over ten seeds. Clinical evaluation fits biomarker coefficients once on the patient-disjoint retrospective biomarker split, freezes them, and applies them unchanged to internal and prospective silent-mode evaluations.

The expected held-out full-model field values are pressure R² 0.981, TAWSS R² 0.972, WSS R² 0.958, OSI R² 0.921, ECAP R² 0.893, PWS R² 0.949 and RPI R² 0.938. Across-seed standard deviations are 0.005, 0.005, 0.006, 0.009, 0.012, 0.007 and 0.008 respectively. TAWSS NMAE is 1.2 percent, pressure NMAE is 0.9 percent, ECAP NMAE is 5.1 percent and PWS NMAE is 3.9 percent.

The coupling-free acceptance values are TAWSS R² 0.970, OSI R² 0.918, PWS R² 0.902 and RPI R² 0.884. PWS NMAE is 6.8 percent and RPI NMAE is 8.1 percent. Flow-field differences must remain within across-seed dispersion while wall-field degradation must remain outside it.

The pooled retrospective biomarker AUC is expected at 0.879 with 95 percent confidence interval 0.846–0.912. Diameter-only AUC is 0.790, peak-wall-stress AUC is 0.856 and rupture-potential-index AUC is 0.868. The prospective confirmatory endpoint requires at least +0.05 diameter-adjusted AUC and positive net reclassification improvement. The observed prospective diameter-adjusted gain is +0.061.

Evaluation must retain pre-specified sex, age-band, site, scanner-vendor and region subgroups. Secondary comparisons use Holm–Bonferroni family-wise correction. Calibration reports intercept, slope and spatial conformal coverage. Abstention reports coverage retained, error among retained cases and false acceptance of excluded morphology.

## Scientific boundaries

The operator is restricted to infrarenal aneurysms with a single dominant sac inside the calibrated diameter, curvature and tortuosity envelope. Suprarenal, post-repair, mycotic and connective-tissue morphologies are outside the modeled geometry class. RPI depends on a population wall-strength model parameterized by ILT thickness, normalized diameter and sex. Wall thickness and material properties are not measured from CTA, so wall-side equilibrium includes bounded population-prior slack.

The prospective arm is silent-mode and non-interventional. Predictions are not intended to alter care. Clinical outcome labels must never enter field-surrogate optimization. All model inputs are pre-event baseline geometries, and all patient and geometry-hash overlaps between pretraining and held-out evaluation must be zero.
