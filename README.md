# AI-Powered Application for Early Detection of Heart Disease Risk

## 📌 Project Overview
This project focuses on building an **AI-powered application for early detection of heart disease risk** using the UCI Heart Disease dataset and OCR-based medical data extraction.  
It includes:
- Data Collection
- Data Cleaning & Preprocessing
- Feature Engineering
- OCR Integration
- Train/Validation/Test Splits

---

## ✅ Project Structure
```
Heart_disease/
├── data/
│ ├── raw/ # Original dataset(s)
│ ├── processed/ # Cleaned, transformed data, splits
│ └── ocr_samples/ 
├── docs/ 
├── outputs/ 
├── scripts/ 
│ ├── data_collection.py
│ ├── eda.py
│ ├── data_cleaning.py
│ ├── feature_engineering.py
│ ├── data_pipeline.py 
│ ├── ocr_pipeline.py
│ ├── ocr_integration.py
│ ├── ocr_accuracy_test.py
├── requirements.txt
└── README.md

```

## ⚙️ Setup Instructions

### **1. Clone the Repository**
```bash
git clone https://github.com/bhargavr004/HEART_DISEASE.git
cd HEART_DISEASE
```
### **2. Create Virtual Environment**
```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```
📂 Dataset

We use the UCI Heart Disease dataset:

Place it in:
```
data/raw/heart_raw.csv
```

🧩 Workflow (Day-wise Tasks)
✅ Day 1: Environment Setup & Data Collection

Set up Python environment, install dependencies

Download UCI Heart Disease dataset → data/raw/

✅ Day 2: Dataset Analysis & EDA

Run:
```
python scripts/eda.py
```

Outputs:

EDA report: outputs/eda_profile_report.html

Visualizations: outputs/figures/

✅ Day 3: Data Cleaning

Run:
```
python scripts/data_cleaning.py
```

Output:

Cleaned data: data/processed/heart_cleaned.csv

✅ Day 4: OCR Pipeline

Run OCR on sample reports:
```
python scripts/ocr_pipeline.py --image data/ocr_samples/sample.png
```
✅ Day 5: Feature Engineering

Run:
```
python scripts/feature_engineering.py
```

Output:

Transformed dataset: data/processed/heart_features.csv

Feature importance: outputs/feature_importances.csv

✅ Day 6: Full Data Pipeline

Run everything end-to-end:
```
python scripts/data_pipeline.py
```

Outputs:

Final train/val/test splits: data/processed/

Milestone report: docs/milestone1_report.md

🧪 OCR Accuracy Test

If you want to test OCR performance:
```
python scripts/ocr_accuracy_test.py
```
Update GROUND_TRUTH in the script for your sample image.

🛠 Tech Stack

Python 3.8+

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, pytesseract, OpenCV

OCR Engine: Tesseract

Version Control: Git, GitHub

📜 License

This project is for educational and research purposes only.
