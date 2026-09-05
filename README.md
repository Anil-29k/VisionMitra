# VisionMitra
## Explainable AI for Diabetic Retinopathy Screening in Rural India

**SIH 2026 Problem Statement:** 26038 / SIH26038  
**Team:** Hack Elite  
**Theme:** MedTech / BioTech / HealthTech

## Overview
VisionMitra is an AI-assisted retinal screening system for rural healthcare settings. The planned pipeline is: Fundus Image → Quality Assessment → Enhancement/Reject → Retinal & Lesion Analysis → DR Grading 0–4 → Grad-CAM/XAI → Annotated Report → Simulink Workflow Simulation.

## Core Goals
- Assess fundus-image quality and provide recapture feedback.
- Enhance borderline images using CLAHE, illumination normalization and denoising.
- Analyze retinal structures and DR lesions.
- Grade DR from Level 0–4 and identify referable DR (Level 2+).
- Provide Grad-CAM, lesion evidence and confidence.
- Generate an annotated screening report.
- Simulate district-level screening capacity in Simulink.

## Technology
- Python: ML/model development, inference, explainability and computer vision.
- MATLAB: image processing, medical-image analysis and evaluation/integration.
- Simulink: acquisition, bandwidth, processing, review capacity and resource simulation.
- Planned Python stack: PyTorch, OpenCV, NumPy, pandas, scikit-learn, Matplotlib and suitable XAI tooling.

## Repository Structure
```text
VisionMitra/
├── data/{raw,processed,test}
├── models/
├── python/{preprocessing,quality,training,inference,explainability,lesions}
├── matlab/{preprocessing,quality,analysis,evaluation}
├── simulink/
├── reports/
├── demo/
├── README.md
├── PROJECT_REQUIREMENTS.md
└── PROJECT_STATUS.md
```

## Development Strategy
First build a working vertical slice: Image → Quality → Preprocessing → DR Model → Grad-CAM → Result. Then expand with lesion analysis, reporting, UI, Simulink and deeper validation.
