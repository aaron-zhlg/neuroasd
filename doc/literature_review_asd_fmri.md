# ASD Classification from Resting-State fMRI: Literature Review

Compiled October 2026. Purpose: give the neuroasd project a reference point — which datasets others use, how they validate, how they train, what they get, and which findings should shape our next steps.

## Key takeaways

- **When tested on truly unseen sites, the credible state of the art is about 65–67% accuracy and 0.72–0.81 AUC.** The strict leave-one-site-out accuracies are Abraham 2017 (66.8%), Heinsfeld 2018 (65%), and a 2023 ABIDE I+II harmonization study (60.7–62.7%). For AUC, IMPAC reached 0.81 on a single unseen site and 0.72 on the external EU-AIMS cohort.
- **The frequently cited "~70% accuracy" numbers are not cross-site.** Parisot 2018 (70.4%), BrainNetTF (71.0%), BrainMass (72.8%), Dong 2025 (72.2%) and Heinsfeld's 70% all come from random or site-stratified splits where every test site is also seen in training. Papers reporting 80%+ go further: random splits plus test-set epoch selection or feature-selection leakage.
- **More data helps, and there is direct evidence for it.** Abraham 2017 and the IMPAC challenge (Traut 2022) both show learning curves that have not saturated at ~2,000 subjects.
- **For cross-site generalization, simple models often beat deep ones.** Of 146 IMPAC submissions, the ones that generalized best combined tangent-space connectivity with linear models (logistic regression / linear SVM). Deep learning and graph-convolutional submissions scored high on the public set and dropped sharply on the private set.
- **Site harmonization (ComBat etc.) helps little.** Under leave-one-site-out, a well-tuned baseline already learns site-invariant features.

## 1. The validation protocol decides the score

| Protocol | Meaning | Score inflation |
|---|---|---|
| Random k-fold / random split | Subjects from the same site can appear in both train and test | High; the model can exploit site signatures |
| Site-stratified random split | Each site is split proportionally into train/val/test | Still high, for the same reason |
| Leave-one-site-out (LOSO) | The test site never appears in training | Close to real deployment |
| External dataset | An independent cohort | Strictest |

Two other common ways scores get inflated:

- **Selecting the best epoch on the test set.** Our project forbids this and reports only the final epoch.
- **Selecting features on the full dataset before cross-validation.** Rosenblatt et al. (Nature Communications 2024) measured this "feature leakage" systematically: it can lift a near-chance prediction task to moderate performance.

## 2. Paper-by-paper summaries

### 2.1 Classic baselines

**Nielsen et al., 2013, Frontiers in Human Neuroscience** — *Multisite functional connectivity MRI classification of autism: ABIDE results*

- Data: ABIDE I, 964 subjects, 16 sites.
- Validation: leave-one-subject-out.
- Method: connections among 7,266 ROIs, filtered by group-difference p-value, then combined as a weighted average.
- Results: 60.0% accuracy (sensitivity 62%, specificity 58%); using all 26.4 million connections gave only 55.7%.
- Takeaway: the first large multi-site baseline. It established "about 60%" as the starting point for multi-site ASD classification.

**Abraham et al., 2017, NeuroImage** — *Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example*

- Data: ABIDE I, 871 subjects (C-PAC preprocessing, after QC), 20 sites.
- Validation: leave-one-site-out, plus several subsets and split schemes.
- Method: a systematic comparison of the whole pipeline:
  - atlases: anatomical vs data-driven (e.g. MSDL);
  - connectivity measures: correlation, partial correlation, tangent space;
  - classifiers: L1/L2 logistic regression, SVM, ridge.
- Results: the best combination (MSDL regions + tangent space + L2 linear classifier) reached 66.8% (≈67%) under LOSO.
- Key findings:
  1. **Accuracy rises with sample size and had not saturated at the full sample.**
  2. Data-driven functional atlases beat standard anatomical atlases.
  3. Tangent-space embedding beats plain and partial correlation.
- Takeaway: the high-quality baseline closest to our setup (same 871-subject C-PAC ABIDE I, same LOSO protocol). 67% is a credible reference line.

**Heinsfeld et al., 2018, NeuroImage: Clinical** — *Identification of autism spectrum disorder using deep learning and the ABIDE dataset*

- Data: ABIDE I, 1,035 subjects, CC200 atlas.
- Validation: 10-fold cross-validation and leave-one-site-out.
- Method: two denoising autoencoders for pretraining, then a fully connected network fine-tuned on top; compared against SVM and random forest.
- Results:
  - 10-fold CV: 70% (sensitivity 74%, specificity 63%);
  - LOSO: 65% (per-site 63–68%);
  - SVM 65%, random forest 63%.
- Caveat: the LOSO results are unusually uniform across sites (even sites with a few dozen subjects fall within 0.63–0.68). Cite with care.
- Takeaway: random splitting versus LOSO costs about 5 percentage points.

**Dvornek et al., 2017, MLMI** — *Identifying Autism from Resting-State fMRI Using LSTM Networks*

- Data: the full ABIDE I.
- Validation: cross-validation (not LOSO).
- Method: an LSTM fed directly with ROI time series.
- Result: 68.5%.

### 2.2 Graph neural networks and Transformers

**Parisot et al., 2018, Medical Image Analysis** — *Disease prediction using graph convolutional networks: application to ASD and Alzheimer's disease*

- Data: ABIDE I, 871 subjects (same as Abraham), Harvard-Oxford 111 ROIs. **This is exactly the setup of our `data/abide`.**
- Validation: stratified 10-fold cross-validation (not LOSO).
- Method: a **population graph**.
  - Each subject is a node.
  - Node features are Fisher-z-transformed connectivity vectors, reduced by ridge-based feature selection.
  - Edge weights come from phenotypic similarity (sex, site) and feature similarity.
  - The setting is semi-supervised / transductive: test subjects sit in the graph, without labels.
- Results: 70.4% accuracy, AUC 0.75.
- Caveat: the graph edges use site information and the setting is transductive, so the numbers are not directly comparable to our LOSO setup.
- Takeaway: Fisher z-transform, feature selection, and phenotypic information are common sources of gains.

**Kan et al., 2022, NeurIPS** — *Brain Network Transformer (BrainNetTF)*

- Data: ABIDE (1,009 subjects, CC200 atlas); ABCD (sex prediction, 7,901 subjects).
- Validation: site-stratified 70/10/20 random split, averaged over 5 runs; the epoch is picked by validation AUROC.
- Method: a Transformer with connection profiles as node features, plus an orthonormal clustering readout (OCREAD).

| Model (ABIDE) | AUROC | Accuracy |
|---|---|---|
| BrainNetTF | 80.2 | 71.0% |
| BrainNetCNN | 74.9 | 67.8% |
| BrainGB | 69.7 | 63.6% |
| BrainGNN | 62.4 | 59.4% |

- Key finding: the authors state that random splitting on ABIDE makes training unstable and opens a large validation–test gap, which is why they stratify by site.
- Takeaway: these are same-site generalization scores, not cross-site. Complex GNNs such as BrainGNN reach only ~59% on ABIDE.

**Cui et al., 2022, IEEE TMI** — *BrainGB: A Benchmark for Brain Network Analysis with GNNs*

- Data: the benchmark experiments use HIV, PNC, PPMI, and ABCD. For ABIDE, only preprocessing scripts are provided.
- Method: GNNs decomposed into four modules (node features, message passing, attention, pooling) and compared systematically.
- Findings:
  1. **Connection profiles** (each ROI's row of the connectivity matrix) work best as node features.
  2. Node-concat message passing works best.
  3. Attention usually helps.
  4. **Concat pooling**, which keeps every node's representation, beats mean or sum pooling.
- Takeaway: our SimpleGCN uses global mean pooling; BrainGB suggests trying concat pooling.

**Dong et al., 2025, Human Brain Mapping** — *A framework for comparison and interpretation of machine learning classifiers to predict autism on the ABIDE dataset* (King's College London)

- Data: ABIDE I (871 subjects).
- Validation: 5-fold CV and 5-fold nested CV, with strict leakage control (feature selection inside each fold).
- Method: reproduced five published models, including Parisot's GCN.
- Results:
  - across models: 58–72% accuracy, AUC 0.64–0.78, with no significant differences between models;
  - best: GCN + majority-vote ensemble (72.2%, AUC 0.77);
  - functional-connectivity-only models averaged 67%; structural-MRI-only models averaged 62%.
- Takeaway: once leakage is removed, very different models converge to ~70%. The bottleneck is the data, not the architecture. Ensembling (majority vote) helped consistently.

### 2.3 Large-scale data and cross-site generalization

**Traut et al., 2022, NeuroImage** — *Insights from an autism imaging biomarker challenge (IMPAC)* — **the most relevant paper for us**

- Data: ABIDE I + II plus an unpublished Robert Debré Hospital (RDB) site, 2,117 subjects in total (947 ASD / 1,170 controls).
  - Public set: 1,150 subjects, ABIDE only.
  - Private test set: 967 subjects (ABIDE + RDB), hidden from participants.
  - After the challenge, EU-AIMS LEAP was used as an external test.
- Data provided:
  - time series for several atlases (BASC, Craddock, Harvard-Oxford, MSDL, Power), plus structural MRI measures;
  - **no subjects excluded for QC**; QC scores were supplied instead.
- Validation: blind evaluation, 146 teams.
- Results:
  - blend of the top 10 submissions: AUC 0.80 on the whole private set. **This is not a leave-one-site-out number**: most private-set subjects are ABIDE participants from sites that also appear in the public training set;
    - as a screening test: 88% sensitivity at a 50% false-positive rate;
    - as a confirmatory test: 25% sensitivity at a 3% false-positive rate;
  - split by origin: median AUC 0.81 on the RDB subjects (the only truly unseen site) versus 0.77 on the ABIDE subjects of the private set;
  - external EU-AIMS (several unseen sites): AUC 0.72 (the authors suspect IQ distribution differences);
  - the paper reports AUC only, no accuracy;
  - fMRI alone AUC 0.79; structural MRI alone 0.66.
- Learning curve: **not saturated at 2,000 subjects**; extrapolation gives AUC ≈0.83 at 10,000 subjects.
- What the winners had in common:
  1. **All top-10 submissions used tangent-space connectivity matrices.**
  2. Most used several atlases combined by stacking.
  3. First-level classifiers were logistic regression or linear SVM.
  4. fMRI and structural MRI were fused with a final logistic regression.
- The authors write that deep-learning submissions, graph convolutions included, showed "strong overfits": public-set AUCs above 0.8 never generalized, while "conservative" submissions in the 0.6–0.8 range stayed stable.
- Takeaways:
  1. The combined ABIDE I + II dataset we just built is the right direction.
  2. **For unseen sites, the realistic reference is AUC 0.72–0.81** at the ~2,000-subject scale with mostly linear models: 0.81 on one unseen site, 0.72 on an external multi-site cohort.
  3. Our GNN should be benchmarked head-to-head against a tangent + logistic regression baseline.

**Ingalhalikar et al., 2021, IEEE TBME** — *Functional Connectivity-Based Prediction of Autism on Site Harmonized ABIDE Dataset*

- Data: ABIDE (ISMRM abstract version: 432 ASD / 556 controls, 18 sites), DPARSF preprocessing, CC200 atlas.
- Validation: leave-one-site-out.
- Method: ComBat harmonization of site differences, then an artificial neural network, random forest, and autoencoders.
- Results: after ComBat, the neural network's LOSO accuracy rose by ~4.5 points; AUROC ≈0.80.
- Caveat: the sources we checked do not establish that ComBat was fit on training sites only. If it was fit on all data including the held-out site, that is a leak (see the next paper) and this LOSO AUC would be optimistic.

**Effect of data harmonization of multicentric dataset in ASD/TD classification, 2023, Brain Informatics**

- Key point: ComBat must be fit **only on training data** (ideally only on the training controls), inside the cross-validation loop. Fitting on the full data inflates results.

**Harmonization techniques for machine learning studies using multi-site fMRI data, bioRxiv 2023**

- Data: ABIDE I + II (1,028 ASD / 1,141 controls, 29 institutions), restricted to the 9 largest sites (N > 50 each); Rest-Meta-MDD as a second dataset.
- Validation: leave-one-site-out and site-stratified k-fold.
- Results:
  - ComBat, CovBat and similar methods pushed site-classification accuracy to near chance;
  - but they **barely changed disorder classification**: LOSO accuracy was 62.7% for the baseline, 60.7% with ComBat, 62.2% with CovBat;
  - the baseline model had already learned site-invariant features.
- Takeaway: consistent with the limited gains we saw from site-alignment tricks (CORAL, adversarial training).

**Holiga et al., 2019, Science Translational Medicine** — *Patients with ASD display reproducible functional connectivity alterations*

- Data: EU-AIMS LEAP (discovery), with ABIDE I, ABIDE II, and InFoR as replication cohorts.
- Method: group-level statistics on degree centrality (not a classifier).
- Results:
  - prefrontal, parietal, and cingulate hyperconnectivity replicated in all three replication cohorts;
  - sensory-motor hypoconnectivity replicated in ABIDE I and InFoR but not ABIDE II.
- Authors' conclusion: the effect sizes are large, but **overlap with controls is also large**, so these alterations are not yet usable as a diagnostic readout.
- Takeaway: a biological explanation for why individual-level classification struggles to go much beyond ~70%.

### 2.4 Pretraining and foundation models

**Yang et al., 2024, BrainMass (arXiv 2403.01433; IEEE TMI)**

- Pretraining data: 30 datasets, 46,686 participants, 70,781 scans. Part of ABIDE I was in pretraining; ABIDE II was used only as an external test.
- Method:
  - a Transformer encoder trained with masked-ROI modeling plus latent-representation alignment;
  - augmentation: drop random time points and recompute FC ("pseudo-functional connectivity").
- Validation: site-stratified 70/15/15 random split.
- Results: 72.75% on ABIDE I (~1.7 points above BrainNetTF); also better than BrainNetTF on ABIDE II.
- Takeaway: **dropping random time points and recomputing FC** is a cheap, effective augmentation. We can use it as long as we keep the time series.

**Brain-JEPA (Dong et al., NeurIPS 2024)**

- Pretrained on 40,162 UK Biobank participants. Downstream tasks are HCP-Aging, ADNI, and others; **the original paper has no ABIDE results**.
- A third-party benchmark (LCM, arXiv 2510.18910) reports 82% on ABIDE, but with a 9.6% standard deviation and a different split. Do not cite this number directly.

**Brain Harmony (NeurIPS 2025)**

- Pretrained on UK Biobank + ABCD; fuses structural and functional MRI.
- Validation: 6:2:2 random split (stratified by site, following BrainNetTF), averaged over 3 runs.

| Model | ABIDE I accuracy | ABIDE II accuracy |
|---|---|---|
| BrainNetCNN | 60.5% | 59.7% |
| BrainGNN | 56.7% | 58.7% |
| BrainNetTF | 56.7% | 62.0% |
| BrainMass | 65.6% | 59.4% |
| BrainHarmonix (multimodal) | 63.1% | 66.7% |

- Takeaways:
  - Reproduced independently, BrainNetTF reaches only 56.7% on ABIDE I, far below the 71% in its own paper. Reported numbers are highly sensitive to data processing and splits.
  - Even the newest large models score only 60–67% on ABIDE.

## 3. Summary table

| Paper | Data | N | Validation | Method | Accuracy | AUC |
|---|---|---|---|---|---|---|
| Nielsen 2013 | ABIDE I | 964 | Leave-one-subject-out | Weighted connections | 60.0% | – |
| Abraham 2017 | ABIDE I | 871 | **LOSO** | MSDL + tangent + L2 linear | 66.8% | – |
| Heinsfeld 2018 | ABIDE I | 1,035 | 10-fold / **LOSO** | Autoencoder + MLP | 70% / 65% | – |
| Dvornek 2017 | ABIDE I | full set | CV | LSTM | 68.5% | – |
| Parisot 2018 | ABIDE I | 871 | 10-fold (transductive) | Population-graph GCN | 70.4% | 0.75 |
| Kan 2022 | ABIDE | 1,009 | Site-stratified random split | BrainNetTF | 71.0% | 0.80 |
| Dong 2025 (HBM) | ABIDE I | 871 | 5-fold (leakage-free) | 5 models + voting | 58–72% | 0.64–0.78 |
| Traut 2022 (IMPAC) | ABIDE I+II+RDB | 2,117 | Blind private set (mostly seen sites) + **unseen site** + **external cohort** | Tangent + linear ensemble | – | 0.80 overall; **0.81 unseen site; 0.72 EU-AIMS** |
| Ingalhalikar 2021 | ABIDE | 988 | **LOSO** (ComBat fitting scope unclear) | ComBat + ANN | – | ~0.80 (possibly optimistic) |
| Harmonization 2023 | ABIDE I+II (9 sites) | – | **LOSO** | ComBat/CovBat + DL | 60.7–62.7% | – |
| BrainMass 2024 | ABIDE I | ~1,084 | Site-stratified random split | Pretrained Transformer | 72.8% | – |
| Brain Harmony 2025 | ABIDE I / II | – | Site-stratified random split | Pretrained multimodal | 63.1% / 66.7% | – |
| **neuroasd (current)** | ABIDE I (C-PAC) | 884 | **LOSO, final epoch** | GCN + improvements | 64.4% | 0.72 |

## 4. Implications for neuroasd

1. **Our numbers are already near the cross-site ceiling.** With LOSO, final-epoch reporting, and no test-set selection, our 64.4% / AUC 0.72 sits just below Abraham 2017 (66.8%, also LOSO) and inside the 0.72–0.81 AUC range IMPAC reports for unseen data. The 70%+ numbers in the literature come from looser protocols and are not the right comparison.
2. **There is no credible precedent for 80% accuracy across unseen sites.** The strict LOSO record is ~67% accuracy. IMPAC, with 2,000+ subjects and 146 teams, reached AUC 0.81 on one unseen site and 0.72 on an external cohort, and reported no accuracy. Plan with ~65–70% LOSO accuracy as the realistic target.
3. **More data helps, but it needs the right model.** Two high-quality learning curves had not saturated, so combining ABIDE I + II is the right call. As data grows, low-capacity, strongly regularized models tend to be more stable.
4. **Add a tangent + logistic regression baseline.** This is what every IMPAC top-10 submission had in common, and it takes a few dozen lines of nilearn / sklearn. If it matches or beats our GCN under LOSO, the GNN's complexity is not paying off.
5. **Literature-backed changes worth trying on the GNN:**
   - tangent-space or Fisher-z connectivity as input (Abraham 2017; Parisot 2018);
   - concat pooling instead of mean pooling (BrainGB);
   - augmentation by dropping random time points and recomputing FC (BrainMass); this requires keeping the time series;
   - multi-atlas stacking, multi-model ensembles, or majority voting (IMPAC; Dong 2025);
   - structural MRI features and phenotypes such as age and sex (IMPAC: structural MRI alone AUC 0.66, a small gain when fused).
6. **Do not expect much from site harmonization.** If used, fit it inside each fold on training data only.
7. **Keep the evaluation rules as they are:** LOSO, final epoch, feature selection and harmonization only inside the training fold. This is where we are more trustworthy than most papers; do not loosen it to chase scores.

## References

1. Nielsen JA et al. Multisite functional connectivity MRI classification of autism: ABIDE results. *Front Hum Neurosci* 2013;7:599. doi:10.3389/fnhum.2013.00599
2. Abraham A et al. Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example. *NeuroImage* 2017;147:736–745. doi:10.1016/j.neuroimage.2016.10.045
3. Heinsfeld AS et al. Identification of autism spectrum disorder using deep learning and the ABIDE dataset. *NeuroImage: Clinical* 2018;17:16–23. doi:10.1016/j.nicl.2017.08.017
4. Dvornek NC et al. Identifying Autism from Resting-State fMRI Using Long Short-Term Memory Networks. *MLMI* 2017. PMC5669262
5. Parisot S et al. Disease prediction using graph convolutional networks. *Med Image Anal* 2018;48:117–130. doi:10.1016/j.media.2018.06.001
6. Kan X et al. Brain Network Transformer. *NeurIPS* 2022. arXiv:2210.06681
7. Cui H et al. BrainGB: A Benchmark for Brain Network Analysis with Graph Neural Networks. *IEEE TMI* 2022. arXiv:2204.07054
8. Dong et al. A framework for comparison and interpretation of machine learning classifiers to predict autism on the ABIDE dataset. *Hum Brain Mapp* 2025. doi:10.1002/hbm.70190
9. Traut N et al. Insights from an autism imaging biomarker challenge: Promises and threats to biomarker discovery. *NeuroImage* 2022;255:119171. doi:10.1016/j.neuroimage.2022.119171
10. Ingalhalikar M et al. Functional Connectivity-Based Prediction of Autism on Site Harmonized ABIDE Dataset. *IEEE TBME* 2021. PMID 33989150
11. Effect of data harmonization of multicentric dataset in ASD/TD classification. *Brain Informatics* 2023. doi:10.1186/s40708-023-00210-x
12. Harmonization techniques for machine learning studies using multi-site functional MRI data. *bioRxiv* 2023. doi:10.1101/2023.06.14.544758
13. Holiga Š et al. Patients with autism spectrum disorders display reproducible functional connectivity alterations. *Sci Transl Med* 2019;11:eaat9223
14. Rosenblatt M et al. Data leakage inflates prediction performance in connectome-based machine learning models. *Nat Commun* 2024. doi:10.1038/s41467-024-46150-w
15. Yang Y et al. BrainMass: Advancing Brain Network Analysis for Diagnosis with Large-scale Self-Supervised Learning. arXiv:2403.01433 (IEEE TMI 2024)
16. Dong Z et al. Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking. *NeurIPS* 2024. arXiv:2409.19407
17. Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens. *NeurIPS* 2025
18. Di Martino A et al. Enhancing studies of the connectome in autism using the autism brain imaging data exchange II. *Sci Data* 2017;4:170010
