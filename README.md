Streamlit Model Inference App

Turbidity monitoring in agricultural wastewater is critical for operational management, regulatory compliance, and environmental protection, yet conventional nephelometric turbidimeters remain inaccessible to most farms due to high capital requirements and technical complexity.

This project presents a smartphone-based Convolutional Neural Network (CNN) system for real-time turbidity classification in agricultural settings. A ResNet-34 architecture with a custom classification head was trained on 1,272 smartphone-captured water sample images across six turbidity classes (0–17.13, 25–90, 150–450, 600–1200, 1300–2500, >2500 NTU).

The training pipeline incorporated MixUp augmentation, label-smoothing cross-entropy loss, and OneCycleLR scheduling. Model robustness was enhanced through RandAugment, color jittering, and geometric transformations, trained with AdamW optimization and gradient clipping.

The system achieved 92.94% validation accuracy with F1-scores exceeding 88% across all turbidity classes. Field deployment demonstrated inference times under 150 ms on standard smartphones without internet connectivity, with robust performance across varying lighting conditions typical of agricultural environments.

The developed mobile application reduces monitoring costs by 95% compared to traditional methods while maintaining accuracy suitable for regulatory compliance. This approach addresses critical infrastructure gaps in rural communities, enabling frequent water quality assessment essential for sustainable agricultural water management.

Repository Structure
.
├── streamlit_app.py   # Main Streamlit application (inference UI)
├── train_colab.py     # Training script (PyTorch + augmentations + MixUp)
├── best.pth           # Trained PyTorch model checkpoint

Getting Started
1. Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt


If no requirements.txt is included, install manually:

pip install streamlit torch torchvision scikit-learn pillow

Running the App

Start the Streamlit app with:

streamlit run streamlit_app.py


This will open a local web interface in your browser (default: http://localhost:8501
).

Training the Model

The training pipeline is provided in train.py. It includes:

Preprocessing: Gray-world white balance, augmentations (RandAugment, jitter, blur, etc.)

Architecture: ResNet-34 backbone with custom classifier head

Regularization: MixUp, label smoothing, dropout, gradient clipping

Optimizer/Scheduler: AdamW + OneCycleLR

Evaluation: Accuracy + confusion matrix on validation set

Running locally
python train_colab.py


By default, it expects your dataset under:

/content/data/images/


with 212 images per class and filenames containing class names
("ultra", "very", "cloudy", "lightly cloudy", "lightly clear", "clear").

Running in Google Colab

Simply upload train_colab.py and your dataset, then run:

!python train_colab.py


The best model checkpoint will be saved as:

best.pth
