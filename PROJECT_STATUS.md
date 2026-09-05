# VisionMitra — Project Status

**Last updated:** 2026-09-05  
**Team:** Hack Elite  
**SIH PS:** 26038 / SIH26038  
**Status:** 🟡 Under Development

## Legend
- ⬜ Not Started
- 🟡 In Progress
- 🟢 Completed
- 🔵 Needs Validation / Improvement
- 🔴 Blocked

## Current Status
| Area | Status |
|---|---|
| Project structure | 🟡 In Progress |
| Python virtual environment | 🟢 Completed |
| Dependencies | ⬜ Not Started |
| Dataset setup | ⬜ Not Started |
| Image input | ⬜ Not Started |
| Quality assessment | ⬜ Not Started |
| Enhancement | ⬜ Not Started |
| Retinal structure analysis | ⬜ Not Started |
| Lesion analysis | ⬜ Not Started |
| DR grading model | ⬜ Not Started |
| Grad-CAM / XAI | ⬜ Not Started |
| Confidence calibration | ⬜ Not Started |
| Dashboard/report | ⬜ Not Started |
| MATLAB integration | ⬜ Not Started |
| Simulink workflow | ⬜ Not Started |
| Validation | ⬜ Not Started |
| Final demo | ⬜ Not Started |
| PPT/demo story | ⬜ Not Started |

## Phase Checklist
### Phase 0 — Setup
- 🟢 Python venv created
- 🟡 Project documentation created
- ⬜ Install dependencies
- ⬜ Initialize Git
- ⬜ Organize dataset
- ⬜ Process first test image

### Phase 1 — Image Pipeline
- ⬜ Image loader
- ⬜ Focus/sharpness score
- ⬜ Illumination check
- ⬜ Field-of-view check
- ⬜ Quality score
- ⬜ Gradeable/ungradeable decision
- ⬜ Recapture feedback
- ⬜ CLAHE
- ⬜ Illumination normalization
- ⬜ Denoising

### Phase 2 — Retinal & Lesion Analysis
- ⬜ Optic disc localization
- ⬜ Fovea localization
- ⬜ Vessel segmentation
- ⬜ Microaneurysm analysis
- ⬜ Exudate analysis
- ⬜ Hemorrhage analysis
- ⬜ Neovascularization analysis
- ⬜ Lesion visualization/evidence

### Phase 3 — DR AI
- ⬜ Dataset preprocessing
- ⬜ Train/validation/test split
- ⬜ Baseline model
- ⬜ Training pipeline
- ⬜ Level 0–4 inference
- ⬜ Referable DR inference
- ⬜ Model export

### Phase 4 — XAI
- ⬜ Grad-CAM
- ⬜ Heatmap overlay
- ⬜ Lesion evidence integration
- ⬜ Confidence scores
- ⬜ Calibration
- ⬜ Human-readable explanation

### Phase 5 — UI/Report
- ⬜ Upload screen
- ⬜ Quality result
- ⬜ Enhancement preview
- ⬜ DR result
- ⬜ Grad-CAM/evidence view
- ⬜ Review workflow
- ⬜ Automated report

### Phase 6 — MATLAB/Simulink
- ⬜ MATLAB integration
- ⬜ Patient arrival model
- ⬜ Acquisition
- ⬜ Bandwidth
- ⬜ AI processing
- ⬜ Ophthalmologist review
- ⬜ Queue/capacity
- ⬜ 100,000+ patient scenario
- ⬜ Optimization results

### Phase 7 — Validation
- ⬜ APTOS
- ⬜ IDRiD
- ⬜ DRIVE
- ⬜ Messidor-2
- ⬜ Sensitivity/specificity
- ⬜ Confusion matrix
- ⬜ F1/ROC-AUC
- ⬜ Cross-dataset testing
- ⬜ Quality-gate evaluation
- ⬜ XAI evaluation

## Immediate Next Actions
1. Install dependencies in `.venv`.
2. Create the full folder structure.
3. Select/download a small initial fundus dataset.
4. Load and display the first image.
5. Build the first image-quality baseline.

## First Milestone
```text
Fundus Image → Quality Check → Preprocessing → DR Model → Grad-CAM → Prediction + Explanation
```

## Change Log
### 2026-09-05
- Development started.
- Python virtual environment created.
- Initial architecture defined.
- README, requirements and status tracker created.
