# VisionMitra — Project Status

**Last updated:** 2026-09-06  **Time**:7.45AM
**Team:** Hack Elite  
**SIH PS:** 26038 / SIH26038  
**Status:** 🟡 Under Development

## Legend

- ⬜ Not Started
- 🟡 In Progress
- 🟢 Completed
- 🔵 Needs Validation / Improvement
- 🔴 Blocked

---

## Current Status

| Area | Status |
|---|---|
| Project structure | 🟢 Completed |
| Python virtual environment | 🟢 Completed |
| Dependencies | 🟢 Completed |
| Dataset setup | 🟢 Completed |
| Image input | 🟢 Completed |
| Quality assessment | 🟢 Completed |
| Enhancement | 🟢 Completed |
| Retinal structure analysis | ⬜ Not Started |
| Lesion analysis | ⬜ Not Started |
| DR grading model | 🔵 Needs Validation / Improvement |
| Grad-CAM / XAI | ⬜ Not Started |
| Confidence calibration | ⬜ Not Started |
| Dashboard/report | ⬜ Not Started |
| MATLAB integration | ⬜ Not Started |
| Simulink workflow | ⬜ Not Started |
| Validation | 🔵 In Progress |
| Final demo | ⬜ Not Started |
| PPT/demo story | ⬜ Not Started |

---

# Phase Checklist

## Phase 0 — Setup

- 🟢 Python venv created
- 🟢 Project documentation created
- 🟢 Dependencies installed
- 🟡 Git initialization
- 🟢 APTOS 2019 dataset organized
- 🟢 Initial fundus image loaded and tested

## Phase 1 — Image Pipeline

- 🟢 Image loader
- 🟢 Focus/sharpness score
- 🟢 Illumination check
- 🟢 Field-of-view check
- 🟢 Quality classification
- 🟢 Gradeable/poor-quality decision
- 🟢 Recapture/reject feedback
- 🟢 CLAHE
- 🟢 Illumination normalization
- 🟢 Denoising
- 🟢 Offline preprocessing pipeline
- 🟢 Processed-image manifest
- 🔵 Quality/enhancement thresholds need further validation

### APTOS Preprocessing

**3662 labeled images processed successfully.**

Quality results:

- GOOD: **3164**
- BORDERLINE: **496**
- POOR: **2**
- Processing failures: **0**

Processed manifest:

```text
data/processed/APTOS-19/manifest.csv
````

Traning Log:

```text
Model_report.txt
```