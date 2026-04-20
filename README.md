# Tumor Synthesis — JHU CV Final Project

3D CT tumor synthesis via copy-paste blending (Hu et al., CVPR 2023) + downstream segmentation with a 3D U-Net.  
Supports Apple Silicon (MPS), CPU, and Google Colab (GPU).

**Team:** Emily Kim, Yoonseo Bae  
**Reference:** Qixin Hu et al., *Label-free Liver Tumor Segmentation*, CVPR 2023

---

## What this does

1. **Copy-paste synthesis** — extracts real tumor patches from labeled CT scans and blends them into healthy volumes using Gaussian feathering
2. **Segmentation training** — trains a 3D U-Net on the synthetic (+ optional real) data
3. **Evaluation** — Dice score, tumor-wise sensitivity, size-stratified sensitivity (small / medium / large)
4. **Low-annotation ablation** — measures how performance changes as the number of real labeled volumes increases

---

## Data: MSD Pancreas (Task07)

This project uses the [Medical Segmentation Decathlon](http://medicaldecathlon.com) pancreas dataset.

- 281 CT volumes with pancreas + tumor labels
- Label values: `1` = pancreas, `2` = tumor
- Free to download (registration not required)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/emilytykim/tumor-synthesis.git
cd tumor-synthesis
```

### 2. Install dependencies

```bash
pip install torch torchvision
pip install monai monai-generative
pip install nibabel scipy matplotlib
```

> Apple Silicon (M1/M2/M3): MPS is used automatically — no extra steps needed.

### 3. Download the data

Go to http://medicaldecathlon.com and download **Task07_Pancreas**.  
Or use the direct Google Drive link listed on that page.

```bash
# After downloading, unzip:
unzip Task07_Pancreas.tar      # or .tar.gz depending on what you downloaded
```

### 4. Set up the data folder

Move the image and label files into this exact structure:

```
tumor-synthesis/
└── data/
    └── raw/
        └── pancreas/
            ├── images/
            │   ├── pancreas_001.nii.gz
            │   ├── pancreas_002.nii.gz
            │   └── ...
            └── labels/
                ├── pancreas_001.nii.gz
                ├── pancreas_002.nii.gz
                └── ...
```

Run these commands to create the folders and move files:

```bash
mkdir -p data/raw/pancreas/images
mkdir -p data/raw/pancreas/labels

# Replace the path below with wherever you unzipped the dataset
cp /path/to/Task07_Pancreas/imagesTr/*.nii.gz data/raw/pancreas/images/
cp /path/to/Task07_Pancreas/labelsTr/*.nii.gz data/raw/pancreas/labels/
```

### 5. Run the notebook

Open VS Code, then open the notebook for your environment:

| Environment | Notebook |
|---|---|
| MacBook (no GPU needed) | `notebooks/tumor_synthesis_macbook.ipynb` |
| Google Colab (T4 GPU) | `notebooks/tumor_synthesis_colab.ipynb` |

Run cells **top to bottom in order**. The notebook will:
- Detect your device (MPS / CUDA / CPU) automatically
- Generate 100 synthetic tumor volumes
- Train the segmenter
- Print Dice + sensitivity results
- Run the low-annotation ablation

---

## Output files

After running, results are saved to:

```
outputs/
├── samples/pancreas/
│   ├── images/          ← synthetic CT volumes (.nii.gz)
│   ├── labels/          ← synthetic tumor masks (.nii.gz)
│   └── previews/        ← PNG visualizations
├── checkpoints/
│   └── segmenter_macbook_best.pth
└── eval_macbook.json    ← Dice + sensitivity numbers
```

---

## Project structure

```
tumor-synthesis/
├── configs/             # YAML configs (base, colab)
├── data/
│   ├── ct_dataset.py    # datalist builders
│   ├── data_module.py   # DataLoader factories
│   └── transforms.py    # MONAI transform pipelines
├── models/
│   ├── autoencoder.py   # 3D VQGAN / AutoencoderKL
│   ├── diffusion.py     # Conditional LDM (UNet + DDPM/DDIM)
│   └── segmentation.py  # 3D U-Net segmenter
├── utils/
│   ├── metrics.py       # Dice, sensitivity, stratified
│   └── nifti_io.py      # NIfTI load/save helpers
├── notebooks/
│   ├── tumor_synthesis_macbook.ipynb
│   └── tumor_synthesis_colab.ipynb
└── scripts/
    └── smoke_test.py    # unit tests for all modules
```
