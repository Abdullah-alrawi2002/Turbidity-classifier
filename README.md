# Streamlit Model Inference App

Turbidity monitoring in agricultural wastewater is critical for operational management, regulatory compliance, and environmental protection, yet conventional nephelometric turbidimeters remain inaccessible to most farms due to high capital requirements and technical complexity.  

This project presents a **smartphone-based Convolutional Neural Network (CNN) system** for real-time turbidity classification in agricultural settings. A **ResNet-34 architecture** with a custom classification head was trained on **1,272 smartphone-captured water sample images** across six turbidity classes (0–17.13, 25–90, 150–450, 600–1200, 1300–2500, >2500 NTU).  

The training pipeline incorporated **MixUp augmentation**, **label-smoothing cross-entropy loss**, and **OneCycleLR scheduling**. Model robustness was enhanced through **RandAugment**, **color jittering**, and **geometric transformations**, trained with **AdamW optimization** and **gradient clipping**.  

The system achieved **92.94% validation accuracy** with **F1-scores exceeding 88%** across all turbidity classes. Field deployment demonstrated **inference times under 150 ms** on standard smartphones without internet connectivity, with robust performance across varying lighting conditions typical of agricultural environments.  

The developed mobile application reduces monitoring costs by **95% compared to traditional methods** while maintaining accuracy suitable for regulatory compliance. This approach addresses critical infrastructure gaps in rural communities, enabling frequent water quality assessment essential for **sustainable agricultural water management**.

---

## 📂 Repository Structure

streamlit_app.py # Main Streamlit application (inference UI)

train.py # Training script (PyTorch + augmentations + MixUp)

⚠️ **Note on Model File**  
We do **not** include `best.pth` (the trained model checkpoint) in this repository because GitHub has a 100 MB file size limit.  
Instead, you have two options:

1. **Train your own model** using `train.py` and save it as `best.pth`.  
   - Place the trained `best.pth` in the project root.  
   - The Streamlit app (`streamlit_app.py`) will automatically load it.  

2. **Download a pre-trained checkpoint** from an external source if provided (e.g., Google Drive, Hugging Face).  
   - Save the file as `best.pth` in the repo root before running the app.  


