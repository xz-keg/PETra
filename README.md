# PETra
Pretrained Evolutionary Transformer for SARS-CoV-2.

Inference: http://cpredict.cn

Codebase for PETRA, built on Megatron-LM.

"data_preprocess" contains codes for data preprocessing.
"model_training" contains codes for model training and some supplementary codes.
"results" contains experimental results.
"inference" for inferencing the model
"petra_article.pdf", paper introducing petra

model weights:[ on huggingface.](https://huggingface.co/xz-keg/PETRA/)

Independent "Readme.txt" about the usage of data preprocess code and model training code can be found in their directories respectively. 

Performance(for 20250212 model, evaluated on seqs collected after 2025-2-12 and available before 2025-7-16)


---
Baseline:
Bloom estimator: https://github.com/jbloomlab/SARS2-mut-fitness

### Nucleotide mutation prediction results for **PETRA**

We report average and weighted recall @1, 10, and 100.
In weighted measure, sequences are weighted by their representativeness.

| **Method**           | **Average Recall @1** |    **@10** |   **@100** | **Weighted Recall @1** |    **@10** |   **@100** |
| -------------------- | --------------------: | ---------: | ---------: | ---------------------: | ---------: | ---------: |
| Random Guess         |                 0.00% |      0.01% |      0.08% |                  0.00% |      0.01% |      0.08% |
| Bloom                |                 0.45% |      1.50% |      9.15% |                  0.49% |      1.48% |      9.41% |
| **PETRA**            |            **11.34%** | **16.92%** | **22.64%** |              **9.45%** | **14.20%** | **19.72%** |

---

### Spike amino-acid mutation prediction results for **PETRA**

We report average and weighted recall @1 and 10.
In weighted measure, sequences are weighted by their representativeness.

| **Method**           | **Average Recall @1** |    **@10** | **Weighted Recall @1** |    **@10** |
| -------------------- | --------------------: | ---------: | ---------------------: | ---------: |
| Random Guess         |                 0.01% |      0.13% |                  0.01% |      0.13% |
| Bloom                |                 6.26% |     12.63% |                  6.64% |     13.08% |
| **PETRA**            |            **17.84%** | **25.69%** |             **17.10%** | **25.58%** |

---


