import argparse
import cv2
import pytesseract
from PIL import Image
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "ocr_samples"
SAMPLES.mkdir(parents=True, exist_ok=True)

def preprocess_image(img_path):
    img = cv2.imread(r"D:/Heart_disease/data/ocr_samples/sample.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # denoise and enhance contrast
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def image_to_text(img_path):
    proc = preprocess_image(img_path)
    pil = Image.fromarray(proc)
    text = pytesseract.image_to_string(pil)
    return text

def parse_medical_values(text):
    """
    Very simple regex-based parsing that looks for common numeric fields.
    Expand regexes to catch more formats as needed.
    """
    results = {}
    # blood pressure patterns (e.g., 120/80)
    m = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', text)
    if m:
        results['systolic'] = int(m.group(1))
        results['diastolic'] = int(m.group(2))
    # cholesterol (mg/dL)
    m = re.search(r'(cholesterol|chol)\D*(\d{2,4})', text, flags=re.I)
    if m:
        results['cholesterol'] = int(m.group(2))
    # heart rate / hr
    m = re.search(r'(hr|heart rate)\D*(\d{2,3})', text, flags=re.I)
    if m:
        results['heart_rate'] = int(m.group(2))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=False, help="path to input image")
    args = parser.parse_args()
    if args.image:
        p = Path(args.image)
    else:
        # pick a sample if provided
        samples = list(SAMPLES.glob("*"))
        if not samples:
            print("No image provided and no samples in data/ocr_samples/. Place sample images there.")
            return
        p = samples[0]
    text = image_to_text(p)
    print("---- RAW TEXT ----")
    print(text)
    parsed = parse_medical_values(text)
    print("---- PARSED VALUES ----")
    print(parsed)

if __name__ == "__main__":
    main()
