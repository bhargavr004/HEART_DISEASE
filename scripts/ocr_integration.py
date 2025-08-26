import cv2
import pytesseract
from PIL import Image
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "ocr_samples"
OUT_FILE = ROOT / "data" / "processed" / "ocr_results.csv"

def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be opened: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,11,2)
    return thresh

def extract_text(img_path):
    processed_img = preprocess_image(img_path)
    pil = Image.fromarray(processed_img)
    text = pytesseract.image_to_string(pil)
    return text

def parse_medical_values(text):
    """Parse common medical fields: BP, cholesterol, heart rate"""
    results = {'systolic': None, 'diastolic': None, 'cholesterol': None, 'heart_rate': None}

    # Blood pressure: e.g. "120/80"
    bp = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', text)
    if bp:
        results['systolic'] = int(bp.group(1))
        results['diastolic'] = int(bp.group(2))

    # Cholesterol: "cholesterol 180" or "Chol: 200"
    chol = re.search(r'(cholesterol|chol)\D*(\d{2,4})', text, re.I)
    if chol:
        results['cholesterol'] = int(chol.group(2))

    # Heart Rate: "Heart Rate: 70" or "HR 85"
    hr = re.search(r'(heart rate|hr)\D*(\d{2,3})', text, re.I)
    if hr:
        results['heart_rate'] = int(hr.group(2))

    return results

def process_all_images():
    if not SAMPLES_DIR.exists():
        print(f"No OCR samples folder found at {SAMPLES_DIR}")
        return None

    files = list(SAMPLES_DIR.glob("*.*"))
    if not files:
        print(f"No images found in {SAMPLES_DIR}")
        return None

    data = []
    for img_path in files:
        print(f"Processing {img_path.name}")
        text = extract_text(img_path)
        parsed = parse_medical_values(text)
        data.append(parsed)

    df = pd.DataFrame(data)
    df.to_csv(OUT_FILE, index=False)
    print(f"OCR results saved to {OUT_FILE}")
    print(df.head())
    return df

if __name__ == "__main__":
    process_all_images()
