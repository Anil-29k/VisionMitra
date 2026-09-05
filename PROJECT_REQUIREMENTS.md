# VisionMitra — Project Requirements

**Source:** SIH 2026 PS 26038 / SIH26038

## Functional Requirements
1. Fundus image input and validation.
2. Quality assessment: focus, illumination, field of view and overall gradeability.
3. Adaptive enhancement: CLAHE, illumination normalization and denoising.
4. Reject ungradeable images and provide recapture feedback.
5. Retinal analysis: optic disc, fovea and vessels.
6. Lesion analysis: microaneurysms, exudates, hemorrhages and neovascularization.
7. DR grading: International Clinical DR scale, Levels 0–4.
8. Referable DR decision: Level 2+.
9. Explainability: Grad-CAM, lesion evidence and confidence.
10. Automated annotated screening report.
11. Human-in-the-loop ophthalmologist review.
12. Simulink workflow for acquisition, bandwidth, AI processing, review capacity and resource allocation for 100,000+ patients/year.

## Performance Targets
- Referable DR sensitivity: >90%
- Referable DR specificity: >85%
- These are targets and must not be claimed as achieved until experimentally validated.

## Validation
Use the planned public benchmarks: APTOS 2019, IDRiD, DRIVE and Messidor-2. Report sensitivity, specificity, precision, recall, F1, confusion matrix and ROC/AUC where appropriate.

## MVP Priority
### Must Have
- Image input
- Quality gate
- Enhancement
- DR Level 0–4 prediction
- Referable DR decision
- Grad-CAM
- Confidence/evidence display
- Basic screening report
- Working end-to-end demo

### Should Have
- Lesion evidence/detection
- Vessel/optic-disc analysis
- Cross-dataset validation
- Simulink resource simulation

### Extension
- Advanced lesion segmentation
- Sophisticated calibration
- Device/domain adaptation
- Large-scale optimization
- More complete clinical validation

## Safety
The prototype is AI-assisted screening support, not a replacement for ophthalmologist diagnosis.
