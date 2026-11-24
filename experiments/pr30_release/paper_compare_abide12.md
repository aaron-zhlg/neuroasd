1120 subjects (475 ASD), git {'branch': 'autoresearch/recall-objective', 'commit': '54be717'}

| Protocol | Who | ASD recall | Control recall | Accuracy | AUC |
|---|---|---|---|---|---|
| Random 10-fold | **PR #30** | 74.1 ± 6.9 | 73.8 ± 5.8 | 73.9 | 0.819 ± 0.023 |
| | ASD-DiagNet 2019 (arXiv 1904.07577, Table II) | 67.8 | 72.8 | 70.1 |  |
| | Heinsfeld 2018 (as re-run by ASD-DiagNet, Table II) | 69.3 | 61.0 | 65.4 |  |
| 70/10/20 x5, site-stratified | **PR #30** | 72.4 ± 8.1 | 71.5 ± 6.2 | 71.9 | 0.796 ± 0.029 |
| | BrainNetTF 2022 (NeurIPS, Table 1) | 72.5 | 69.3 | 71.0 | 0.802 |
| 70/15/15 x10, site-stratified | **PR #30** | 68.0 ± 6.6 | 75.8 ± 6.6 | 72.5 | 0.804 ± 0.025 |
| | BrainMass 2024 (arXiv 2403.01433, Table II) | 74.26 | 71.89 | 72.75 |  |

ROC read-out: PR #30 ASD recall at the paper's control recall (fold mean)

| Protocol | Paper | Paper ASD recall | PR #30 ASD recall at the same control recall |
|---|---|---|---|
| Random 10-fold | ASD-DiagNet 2019 (arXiv 1904.07577, Table II) (control 72.8) | 67.8 | 76.5 |
| Random 10-fold | Heinsfeld 2018 (as re-run by ASD-DiagNet, Table II) (control 61.0) | 69.3 | 86.3 |
| 70/10/20 x5, site-stratified | BrainNetTF 2022 (NeurIPS, Table 1) (control 69.3) | 72.5 | 74.3 |
| 70/15/15 x10, site-stratified | BrainMass 2024 (arXiv 2403.01433, Table II) (control 71.89) | 74.26 | 74.2 |
