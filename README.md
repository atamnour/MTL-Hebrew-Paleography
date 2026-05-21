# MTL-Hebrew-Paleography

[![Paper](https://img.shields.io/badge/Paper-Springer-blue.svg)](https://link.springer.com/chapter/10.1007/978-3-032-04630-7_5)
[![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/atamnour/MTL-Hebrew-Paleography)
[![Dataset](https://img.shields.io/badge/Dataset-VML--MHS-green.svg)](https://github.com/atamnour/MTL-Hebrew-Paleography/tree/main/dataset_card/VML-MHS)

Official code for the ICDAR 2025 paper:

**Multi-task Learning for Hebrew Paleography: Script Classification and Date Estimation**

---

## Overview

This repository contains the implementation of a multi-task learning framework for medieval Hebrew manuscript analysis.

The model jointly performs:

1. Hebrew script type and mode classification.
2. Manuscript date estimation.

The project uses the **VML-MHS** dataset, which contains medieval Hebrew manuscript images annotated with script type, script mode, and production year.

---

## Paper

Springer link:

[Multi-task Learning for Hebrew Paleography: Script Classification and Date Estimation](https://link.springer.com/chapter/10.1007/978-3-032-04630-7_5)

---

## Dataset

The dataset information is available here:

[VML-MHS Dataset Card](https://github.com/atamnour/MTL-Hebrew-Paleography/tree/main/dataset_card/VML-MHS)

Dataset summary:

| Item | Count |
|---|---:|
| Manuscripts | 2,304 |
| Pages | 3,687 |
| Patches | 346,178 |
| Time span | 850–1540 CE |

---

## Repository Structure

```text
MTL-Hebrew-Paleography/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── HebrewPaleopraphyLoader.py
│   ├── MultiTaskModel.py
│   ├── ModelsConfigsNew.py
│   ├── train.py
│   ├── evaluate_patch_level.py
│   ├── evaluate_page_level.py
│   ├── predict.py
│   └── extract_page_patches.py
│
└── scripts/
    └── data_preparation/
        ├── Step1_CreatDataset.py
        ├── Step2_AnalysisOriginalDataset.py
        ├── Step3_ExtractPatches.py
        ├── Step4_AnalysisPatchesDataset.py
        ├── Step5_GroupDatasetToDecades.py
        └── Step6_WithBlindTest.py