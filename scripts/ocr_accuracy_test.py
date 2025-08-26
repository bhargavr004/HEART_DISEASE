import cv2
import pytesseract
from PIL import Image
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = ROOT / "data" / "ocr_samples" / "sample.png"

# Ground truth for sample.png (update these based on actual image text)
GROUND_TRUTH = {
    "systolic": 120,
    "diastolic": 80,
    "cholesterol": 190,
    "heart_rate": 72
}

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
    results = {'systolic': None, 'diastolic': None, 'cholesterol': None, 'heart_rate': None}

    bp = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', text)
    if bp:
        results['systolic'] = int(bp.group(1))
        results['diastolic'] = int(bp.group(2))

    chol = re.search(r'(cholesterol|chol)\D*(\d{2,4})', text, re.I)
    if chol:
        results['cholesterol'] = int(chol.group(2))

    hr = re.search(r'(heart rate|hr)\D*(\d{2,3})', text, re.I)
    if hr:
        results['heart_rate'] = int(hr.group(2))

    return results

def calculate_accuracy(predicted, ground_truth):
    correct = 0
    total = len(ground_truth)
    for key in ground_truth:
        if predicted[key] == ground_truth[key]:
            correct += 1
    return correct / total * 100

def main():
    print(f"Testing OCR accuracy for: {SAMPLE_IMAGE}")
    text = extract_text(SAMPLE_IMAGE)
    print("Extracted text:\n", text)

    predicted = parse_medical_values(text)
    print("Parsed values:", predicted)

    accuracy = calculate_accuracy(predicted, GROUND_TRUTH)
    print(f"OCR Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
