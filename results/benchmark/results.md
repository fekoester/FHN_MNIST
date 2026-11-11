| method | fashion | mnist | train time (s) |
|---|---|---|---|
| Logistic regression (pixels) | 83.68% | 92.18% | 23/24 |
| MLP 784-512-10 | 87.45% | 97.75% | 36/142 |
| FHN, readout only (BP) | 84.26% | 92.32% | 69/89 |
| FHN, BPTT, W_rec frozen | 85.65% | 97.06% | 110/108 |
| FHN, BPTT, all weights | 86.02% | 96.24% | 125/124 |
| FHN, teacher, readout only | 81.83% | 90.96% | 94/98 |
| FHN, teacher, 3-factor | 83.59% | 95.03% | 122/128 |
| FHN, reward, readout only | 77.24% | 87.36% | 101/104 |
| FHN, reward, 3-factor | 77.40% | 89.72% | 140/127 |
