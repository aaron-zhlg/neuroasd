# External cohorts for a model trained on ABIDE I and II

This note lists datasets that can score a model trained on the DCAN ABIDE I + II set
(`data/abide_dcan`, 1,443 subjects) on people the model has never seen. "HCP-level" here
means the imaging protocol is the Human Connectome Project protocol or a shortened
version of it. It does not mean the cohort is one of the HCP disease studies.

The released scorer lives on `experiment/trial-iter10_estimator-swap-replace-the-singl`
(`scripts/test_model.py`, tag `iter10-model-v1`). It expects Gordon and HCP2016
functional connectivity in the `abide_dcan` layout, plus age and sex. A new cohort is
usable after its resting-state time series are parcellated the same way. Per-group
scoring z-scores a whole site together; per-subject scoring is the one to use when
each person is scored alone.

Leave-one-site-out on ABIDE still trains on other ABIDE scanners. The cohorts below
are a different test: another study, another label definition, another age mix.
IMPAC's drop from AUC 0.81 on one unseen ABIDE site to AUC 0.72 on EU-AIMS LEAP is
the published version of that drop (`doc/literature_review_asd_fmri.md`).

## What to use

| Cohort | Role for this project | Imaging | ASD labels | Access |
|---|---|---|---|---|
| **ABIDE I + II** | Training set already in the repo | Multi-site, protocols differ by site | Clinical ASD vs control | Public, FCP-INDI. DCAN derivatives are what we build from |
| **EU-AIMS LEAP** | First external ASD test | Multi-site Europe, multimodal | ASD cohort, deep phenotyping | Application. Processed data only |
| **HBN** | Pediatric external test, ASD and ADHD | Child and adolescent, some releases open | Community-referred diagnoses, including ASD and ADHD | MRI listed as open downloads; phenotypes need a DUA |
| **ABCD** | ASD-related items and BMI in one HCP-style cohort | Baseline at ages 9–10, longitudinal, protocol close to HCP | Parent-report KSADS category. Moderate/severe ASD was excluded at enrollment | NDA data-use certification |

EU-AIMS LEAP is the external ASD test. HBN is the more open pediatric alternative.
ABCD is the cohort to use when the question includes body mass index. The 14
Connectomes Related to Human Disease projects and HCP-Development are protocol
references; their samples are not ASD case-control sets.

## ABIDE I and II

ABIDE is the main public ASD imaging dataset and the one this repository trains on.
ABIDE I and ABIDE II are pooled after the same DCAN (ABCD-HCP) pipeline, Gordon and
HCP2016 atlases, in `data/abide_dcan/README.md`.

The limitation for an external test is the acquisition itself. Each site kept its own
scanner and protocol, so a leave-one-site-out score still mixes site shift with the
ASD effect. A second cohort with one shared protocol answers a different question.

## EU-AIMS LEAP

The Longitudinal European Autism Project (EU-AIMS, continued as AIMS-2-TRIALS) is a
multi-center European ASD cohort with clinical, cognitive, eye-tracking, EEG, MRI,
and biological measures. Resting-state fMRI is part of the imaging. Holiga et al.
2019 used LEAP as the discovery set and ABIDE I, ABIDE II, and InFoR as replications.
After the IMPAC challenge, Traut et al. 2022 scored the winning models on LEAP and
reported AUC 0.72.

Access is by application. AIMS-2-TRIALS currently shares processed timepoint-1 and
timepoint-2 data through ELIXIR Luxembourg, after a project proposal and ethics
approval for secondary analysis. Raw scans are not in that share. The public
catalogue entry for resting-state fMRI lists on the order of 500 scans at timepoint 1
and about 380 at timepoint 2. Confirm counts in the data dictionary after approval,
because those numbers are scans, not necessarily unique subjects.

This is the closest match to "train on ABIDE, test on another ASD cohort."

- Researcher access: https://www.aims-2-trials.eu/our-research/data-access-researcher/
- Resting-state catalogue: https://datacatalogue.elixir-luxembourg.org/e/dataset/ELU-2-4A5E20-1

## Healthy Brain Network (HBN)

HBN (Child Mind Institute) recruits children and adolescents in New York, ages 5–21,
from families who already have a concern about psychiatric symptoms. The target is
10,000 participants. The protocol is transdiagnostic: ASD and ADHD are both common,
alongside other diagnoses, with MRI, EEG, genetics, and a deep behavioral battery
(including ASSQ, SCQ, and SRS-2). Height, weight, and bio-impedance are in the
physical measures, so body composition can be studied, in a referred sample rather
than a population cohort.

Imaging releases are posted for download from the HBN biobank. Phenotypes require a
data-use agreement, which the project says can take up to a few weeks, and then a
LORIS account. Cite Alexander et al., Scientific Data, 2017.

- Biobank: https://data.healthybrainnetwork.org/main.php
- Data descriptor: https://www.nature.com/articles/sdata2017181

## ABCD

The Adolescent Brain Cognitive Development study follows children from ages 9–10
with a harmonized imaging protocol built from the HCP protocol and shortened for
children. Structural MRI, resting-state fMRI, task fMRI, and diffusion MRI are
collected, and the same children return longitudinally. BMI and the rest of the
physical-health measures are in the tabulated data. The DCAN ABCD-HCP pipeline that
produced our ABIDE derivatives was written for this study, so ABCD is the closest
preprocessing relative of the training set.

Use ABCD when the scientific question needs ASD-related information and obesity in
the same HCP-style dataset. Two facts from the study's own data-access FAQ bound
that use:

- Current moderate or severe autism spectrum disorder was an enrollment exclusion
  (a past diagnosis that has remitted was allowed). Schizophrenia, intellectual
  disability, and alcohol or substance-use disorder were excluded on the same rule.
- The study does not keep a separate clinical ASD record. The KSADS includes an
  autism spectrum disorder category, and that item is parent report.

ABCD is therefore a population cohort with a parent-report ASD item and measured
BMI, scanned like HCP. It is a weak stand-in for ABIDE's clinical ASD-versus-control
contrast, because the moderate and severe cases ABIDE is built on were screened out.

Access is an NDA Data Use Certification. The applicant needs a research institution
with an NIH-recognized signing official. There is no fee. International institutions
with a Federalwide Assurance have been approved. Imaging and tabulated releases,
including Data Release 5.1, are on the NIMH Data Archive.

- Access: https://nda.nih.gov/abcd
- Study FAQ (autism exclusion and KSADS): https://docs.abcdstudy.org/latest/info/faq.html

## HCP sources that are not ASD test sets

These are the pages the HCP-level protocol comes from. They explain what "HCP-style"
means. They do not supply an ASD-versus-control label for this model.

### Connectomes Related to Human Disease (14 projects)

NIH funded 14 groups under the disease-connectome program to scan people with a
specific condition on an HCP-style protocol, so the scans can be compared with
healthy HCP subjects. The Human Connectome Project page and the NIH Blueprint award
table list them as follows. Wikipedia groups the same 14 as four Alzheimer's or
dementia studies, two vision studies, four anxiety or depression studies, two
psychosis studies, one epilepsy study, and the Amish mental-illness study.

| Project | Focus | Ages |
|---|---|---|
| Alzheimer's Disease Connectome Project | Alzheimer's disease | 50–90 |
| The structural and functional connectome across Alzheimer's disease subtypes | Alzheimer's subtypes | 45–80 |
| Connectomics of Brain Aging and Dementia | Brain aging and dementia | 50–89 |
| Connectomic imaging in familial and sporadic frontotemporal degeneration | Frontotemporal degeneration | 21+ |
| Changes in Visual Cortical Connectivity Following Central Visual Field Loss | Central vision loss | 19–89 |
| Human Connectomes for Low Vision, Blindness, and Sight Restoration | Low vision, blindness, sight restoration | 18–80 |
| Dimensional connectomics of anxious misery | Anxious misery | 18–45 |
| Perturbation of the treatment resistant depression connectome by fast-acting therapies | Treatment-resistant depression | 20–65 |
| Connectomes Related to Anxiety and Depression in Adolescents (BANDA) | Adolescent anxiety and depression | 14–17 |
| Mapping connectomes for disordered emotional states | Disordered emotional states | 22–35 |
| Human Connectome Project for Early Psychosis | Early psychosis | 16–30 |
| Neural Disconnection and Errant Visual Perception in Psychotic Psychopathology | Psychosis | 18–59 |
| The Epilepsy Connectome Project | Epilepsy | 18–50 |
| Amish Connectome Project on Mental Illness | Mental illness in Amish families | 18–80 |

No row is an ASD cohort. Released CRHD imaging currently on the Connectome
Coordination Facility includes HCP-Early Psychosis, BANDA, the treatment-resistant
depression project, and Dimensional Connectomics of Anxious Misery.

- Study list: https://www.humanconnectome.org/disease-studies
- Award table: https://neuroscienceblueprint.nih.gov/human-connectome/connectome-projects
- Released sets: https://nda.nih.gov/ccf

### HCP Lifespan Development

HCP-Development (HCP-D) scans healthy children, adolescents, and young adults, ages
5–21, with a shortened HCP protocol. The planned sample is about 1,350. Lifespan
Release 2.0 shares visit-1 data for 652 participants. The study describes that
healthy developmental curves as a reference for conditions that affect wiring,
including autism. The enrolled sample is the healthy reference, so it does not
label ASD cases for a classifier test.

The rest of the lifespan series is the same kind of resource at other ages:
prenatal (Developing Human Connectome Project), 0–5 (Baby Connectome), and 36–100+
(HCP-Aging).

- HCP-D: https://www.humanconnectome.org/study/hcp-lifespan-development/
- Lifespan overview: https://www.humanconnectome.org/

## Sources

- Human Connectome Project disease studies: https://www.humanconnectome.org/disease-studies
- HCP Lifespan Development: https://www.humanconnectome.org/study/hcp-lifespan-development/
- Human Connectome Project, Wikipedia (the 14 CRHD groups): https://en.wikipedia.org/wiki/Human_Connectome_Project
- NIH Blueprint connectome awards: https://neuroscienceblueprint.nih.gov/human-connectome/connectome-projects
- ABCD on the NIMH Data Archive: https://nda.nih.gov/abcd
- ABCD data-access FAQ: https://docs.abcdstudy.org/latest/info/faq.html
- AIMS-2-TRIALS researcher access: https://www.aims-2-trials.eu/our-research/data-access-researcher/
- HBN biobank: https://data.healthybrainnetwork.org/main.php
- Alexander et al. 2017, HBN data descriptor: https://doi.org/10.1038/sdata.2017.181
