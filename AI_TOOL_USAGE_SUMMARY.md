# AI Tool Usage Summary
## CIT Hackathon – PS-07: CT-to-MRI Translation with Reliability Estimation

### 1. AI Tool Used

**ChatGPT (OpenAI)**

ChatGPT was used as an AI-assisted development and problem-solving tool throughout the project development process.

### 2. Areas of Utilization

| Development Area | How ChatGPT Was Utilized |
|---|---|
| Project Planning | Discussed the problem statement, solution workflow, and implementation approach. |
| Dataset Understanding | Assisted in understanding the CT-MRI dataset structure, normalization, preprocessing, and validation strategy. |
| Model Development | Provided guidance on Residual U-Net architecture, Pix2Pix, and PatchGAN integration. |
| Debugging | Assisted in understanding Python errors, notebook issues, model integration problems, and inference errors. |
| Reliability Estimation | Explained Test-Time Augmentation (TTA), prediction variation, uncertainty maps, and evaluation using Spearman correlation. |
| EvidenceMap Integration | Assisted in adapting patch retrieval code to the NIfTI-based dataset and understanding similarity-based retrieval. |
| Model Evaluation | Helped interpret MAE, SSIM, PSNR, Dice scores, and uncertainty-error correlation. |
| Visualization | Assisted in preparing Matplotlib plots to compare model performance and visualize experimental results. |
| Documentation | Assisted in organizing project documentation, README content, and GitHub repository preparation. |

### 3. AI-Assisted Development Workflow

The development process involved using ChatGPT for technical explanations, implementation guidance, debugging suggestions, and interpretation of experimental results.

The project team performed the actual implementation, executed experiments, inspected outputs, evaluated models, and made development decisions based on the observed results.

### 4. Human Verification and Responsibility

- All model training and inference experiments were executed by the project team.
- Dataset preprocessing and validation outputs were inspected.
- Experimental metrics were obtained from the implemented evaluation procedures.
- AI-generated suggestions were reviewed and tested before being incorporated.
- Final implementation choices and project results were verified by the team.

### 5. Project Components Developed

The project includes:

1. Residual U-Net-based CT-to-MRI translation.
2. Pix2Pix with a PatchGAN discriminator.
3. Test-Time Augmentation-based voxel-wise reliability estimation.
4. EvidenceMap-based retrieval of similar CT-MRI patches.
5. Quantitative evaluation using MAE, SSIM, PSNR, Dice score, and Spearman correlation.

### 6. Transparency Statement

ChatGPT served as an AI-assisted development resource for technical guidance, debugging, documentation, and result interpretation. It did not independently conduct the experiments or validate the scientific performance of the proposed system. The project team is responsible for the implementation, experimental verification, and final submission.