# Public fMRI and Neuroimaging Datasets

## 1. Overview

fMRI (functional Magnetic Resonance Imaging) is a **neuroimaging modality**, not a dataset by itself. It measures changes in blood-oxygen-level-dependent (BOLD) signals over time and is commonly represented as a 4D volume:

```text
X × Y × Z × T
```

where `X, Y, Z` represent spatial dimensions and `T` represents time.

Public datasets and repositories such as HCP, ABIDE, ADNI, OpenNeuro, and UK Biobank provide fMRI and/or other neuroimaging data for neuroscience and medical AI research.

A common fMRI-to-AI pipeline is:

```text
fMRI
  ↓
Preprocessing
  ↓
Brain parcellation / ROI extraction
  ↓
Regional time series
  ↓
Functional connectivity
  ↓
Brain graph
  ↓
GNN / Transformer / ML
  ↓
Disease or cognitive prediction
```

---

## 2. HCP — Human Connectome Project

**Type:** Large neuroimaging research project / dataset  
**Primary focus:** Human brain connectivity and cognitive neuroscience  
**fMRI:** Yes

The Human Connectome Project (HCP) is one of the most important resources for studying human brain connectivity. HCP provides structural MRI, resting-state fMRI, task fMRI, diffusion MRI, and behavioral/cognitive measurements.

### Why it is useful

HCP is particularly well suited for:

- Functional connectivity
- Brain network analysis
- Connectomics
- Brain graph learning
- Cognitive prediction
- Representation learning

A typical graph formulation is:

```text
Brain regions → graph nodes
Functional connectivity → graph edges
Subject-level cognition → prediction target
```

### Data access

HCP data are openly available to the scientific community, but access requires registration and agreement to the HCP Data Use Agreement. After approval, users can download the data.

**Official resource:** https://www.humanconnectomeproject.org/data/

### AI research value

**Excellent for:** GNNs, graph transformers, connectome modeling, functional connectivity, and brain-behavior prediction.

---

## 3. ABIDE — Autism Brain Imaging Data Exchange

**Type:** Multi-site neuroimaging dataset  
**Primary focus:** Autism Spectrum Disorder (ASD)  
**fMRI:** Yes, especially resting-state fMRI

ABIDE aggregates neuroimaging data from multiple research sites and is widely used for studying differences between individuals with ASD and healthy controls.

A common research setup is:

```text
Resting-state fMRI
       ↓
Functional connectivity
       ↓
Subject-specific brain graph
       ↓
ASD vs. healthy control
```

### Why it is useful

ABIDE is particularly useful for:

- ASD classification
- Functional connectivity analysis
- Brain network biomarkers
- GNN-based disease prediction
- Multi-site/domain generalization

Because ABIDE combines data from multiple sites, scanner and acquisition differences create substantial cross-site heterogeneity. This makes it useful not only for classification, but also for studying domain adaptation and generalization.

### Data access

ABIDE data are publicly available through the International Neuroimaging Data-sharing Initiative (INDI) ecosystem.

Cohorts that can score a model already trained on ABIDE, including EU-AIMS LEAP, the Healthy Brain Network, and ABCD, are listed in [External test sets](external_asd_test_datasets.md).

### AI research value

**Excellent for:** supervised disease classification, brain graph learning, and multi-site robustness.

---

## 4. ADNI — Alzheimer's Disease Neuroimaging Initiative

**Type:** Large longitudinal multimodal research project  
**Primary focus:** Alzheimer's disease and cognitive decline  
**fMRI:** Available, but ADNI is broader than fMRI

ADNI is particularly important because it combines multiple modalities and longitudinal clinical information.

Depending on the study and participant, ADNI includes:

- Structural MRI
- fMRI
- PET
- Clinical assessments
- Cognitive measurements
- Biomarkers
- Genetic information

A multimodal AI setup can look like:

```text
                 ┌── Structural MRI
                 ├── fMRI
Patient ─────────┼── PET
                 ├── Clinical data
                 ├── Cognitive scores
                 └── Biomarkers
                         ↓
                  Multimodal model
                         ↓
              Disease / progression prediction
```

### Why it is useful

ADNI is especially valuable for:

- Alzheimer's disease classification
- Mild Cognitive Impairment prediction
- Cognitive decline prediction
- Disease progression modeling
- Multimodal medical AI
- Biomarker discovery

### Data access

ADNI data require registration and acceptance of the applicable data-use requirements. It is not a completely unrestricted download repository.

**Official resource:** https://adni.loni.usc.edu/

### AI research value

**Excellent for:** multimodal medical AI and longitudinal disease modeling.

---

## 5. OpenNeuro

**Type:** Public neuroimaging data repository / platform  
**Primary focus:** Open sharing of many different neuroscience datasets  
**fMRI:** Yes, extensively

OpenNeuro is different from HCP, ABIDE, and ADNI: it is **a platform hosting many datasets**, rather than one single dataset.

It hosts datasets containing modalities such as:

- fMRI
- Structural MRI
- EEG
- MEG
- iEEG
- PET

Many datasets are organized according to **BIDS (Brain Imaging Data Structure)**.

A typical dataset may look like:

```text
dataset/
├── sub-001/
│   ├── anat/
│   │   └── sub-001_T1w.nii.gz
│   └── func/
│       └── sub-001_task-rest_bold.nii.gz
├── sub-002/
│   └── ...
└── dataset_description.json
```

### Why it is useful

OpenNeuro is one of the easiest places to start experimenting with real neuroimaging data.

It is particularly useful for:

- Learning fMRI processing
- Building preprocessing pipelines
- Testing ML/GNN models
- Reproducing published experiments
- Finding datasets for specific tasks

### Data access

Public OpenNeuro datasets can be browsed and downloaded without an account.

**Official resource:** https://openneuro.org/

### AI research value

**Excellent for:** prototyping and quickly testing neuroimaging ML pipelines.

---

## 6. UK Biobank

**Type:** Large population cohort / research resource  
**Primary focus:** Population health, genetics, lifestyle, and disease  
**MRI:** Yes  
**fMRI:** Available within its broader imaging resource

UK Biobank is much larger in scale than most neuroscience datasets. It combines imaging with extensive non-imaging information, including genetics, lifestyle, physical measurements, and healthcare-related data.

The imaging program has reached approximately **100,000 participants with imaging data**.

A typical research setup is:

```text
                 ┌── Brain MRI
                 ├── Other imaging
Participant ─────┼── Genetics
                 ├── Lifestyle
                 ├── Clinical data
                 └── Longitudinal information
                         ↓
                    Large-scale AI
                         ↓
              Risk / phenotype prediction
```

### Why it is useful

UK Biobank is particularly valuable for:

- Brain aging
- Dementia research
- Population-level brain analysis
- Genetics × brain imaging
- Disease risk prediction
- Large-scale multimodal AI

### Data access

UK Biobank data are **not freely downloadable without an application**. Researchers must apply for access under UK Biobank's data-access framework.

**Official resource:** https://www.ukbiobank.ac.uk/about-our-data/types-of-data/imaging-data/

### AI research value

**Excellent for:** large-scale multimodal modeling and population-level research.

---

## 7. Comparison

| Resource | Type | fMRI | Main Focus | Access | AI / GNN Value |
|---|---|---:|---|---|---:|
| **HCP** | Research dataset/project | Yes | Brain connectivity | Registration + agreement | ★★★★★ |
| **ABIDE** | Multi-site dataset | Yes | Autism / ASD | Publicly available | ★★★★★ |
| **ADNI** | Longitudinal multimodal project | Yes | Alzheimer's / cognition | Registration + application | ★★★★★ |
| **OpenNeuro** | Dataset repository | Yes | Many neuroscience tasks | **Public datasets can be downloaded directly** | ★★★★☆ |
| **UK Biobank** | Population cohort | Yes / broader MRI | Population health + genetics | Application required | ★★★★★ |

---

## 8. Recommended Starting Point for AI Research

For someone starting a project around **fMRI + brain graphs + AI**, a practical progression is:

### Stage 1 — OpenNeuro

Start with a relatively small public dataset.

Goal:

```text
fMRI
 ↓
Preprocessing
 ↓
ROI extraction
 ↓
Functional connectivity
 ↓
Brain graph
 ↓
GNN
```

This is the fastest way to understand the complete pipeline.

### Stage 2 — ABIDE

Move to a disease classification problem:

```text
fMRI
 ↓
Brain graph
 ↓
GNN
 ↓
ASD vs. Control
```

This provides a concrete supervised learning benchmark.

### Stage 3 — HCP

Focus on brain connectivity and representation learning:

```text
fMRI
 ↓
Functional connectome
 ↓
Graph representation
 ↓
Cognitive prediction
```

### Stage 4 — ADNI

Move toward multimodal medical AI:

```text
MRI + fMRI + PET + clinical data
                    ↓
             Multimodal model
                    ↓
          Alzheimer's prediction
```

### Stage 5 — UK Biobank

Use a much larger population-scale dataset once the methodology is mature.

---

## 9. Key Terminology

It is important not to treat these terms as equivalent:

```text
fMRI
│
└── Neuroimaging modality / data type
       │
       ├── HCP
       ├── ABIDE
       └── ADNI
             └── Specific research datasets/projects

OpenNeuro
└── Repository containing many neuroimaging datasets

UK Biobank
└── Large population cohort containing imaging + genetics + health data
```

In short:

> **fMRI is a data modality. HCP, ABIDE, and ADNI are research datasets/projects. OpenNeuro is a public neuroimaging repository. UK Biobank is a large population cohort with extensive imaging and health data.**

For an initial **AI + neuroscience** project, OpenNeuro is the easiest place to start, ABIDE is a strong benchmark for disease classification, HCP is excellent for connectomics, and ADNI/UK Biobank are particularly valuable for multimodal medical AI.

## 10. Official Resources

- HCP: https://www.humanconnectomeproject.org/data/
- OpenNeuro: https://openneuro.org/
- OpenNeuro Documentation: https://docs.openneuro.org/
- ADNI: https://adni.loni.usc.edu/
- UK Biobank Imaging: https://www.ukbiobank.ac.uk/about-our-data/types-of-data/imaging-data/
