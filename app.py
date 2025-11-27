from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import logging
import os
import cv2
import numpy as np
import easyocr
import requests
import fitz

# --- Basic Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Configuration ---
OLLAMA_API_URL = "http://10.100.61.225:11434/api/generate"
OLLAMA_MODEL_NAME = "gemma3:12b"

# Define a path for models *inside* our project directory
# We look for a 'models' folder in the current directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'easyocr_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Model Initialization ---
try:
    logging.info(f"Loading EasyOCR models from: {MODEL_DIR}")
    # Initialize Reader.
    # Note: We do NOT pass 'download_enabled=True' to prevent it from trying to reach the internet.
    # The models must be present in MODEL_DIR.
    reader = easyocr.Reader(['en', 'ar'], gpu=True, model_storage_directory=MODEL_DIR, download_enabled=False)
    logging.info("EasyOCR models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to initialize EasyOCR. Ensure model files are in {MODEL_DIR}: {e}", exc_info=True)
    reader = None

# --- Preprocessing Functions ---
def preprocess_image_for_ocr(image_bytes):
    """
    Enhances image contrast and binarizes it to help EasyOCR
    distinguish text from complex infographic backgrounds.
    """
    try:
        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is better than global equalization for images with varying lighting/colors
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 3. (Optional) Denoising - useful if the image is grainy
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=10)

        # Encode back to bytes for EasyOCR
        is_success, buffer = cv2.imencode(".png", denoised)
        if is_success:
            return buffer.tobytes()
        else:
            return image_bytes  # Fallback

    except Exception as e:
        logging.warning(f"Image preprocessing failed, using original: {e}")
        return image_bytes

# --- OCR Core Logic ---
def ocr_image_with_easyocr(image_stream_bytes):
    if reader is None:
        return "ERROR: OCR Service is not ready. Models missing?"

    try:
        # Step 1: Preprocess the image
        processed_bytes = preprocess_image_for_ocr(image_stream_bytes)

        # Step 2: Run EasyOCR with Tuned Parameters for Layouts
        # x_ths: Horizontal threshold. Lower value (e.g., 0.1) prevents merging separate columns.
        # y_ths: Vertical threshold. Higher value (e.g., 0.5-0.7) helps merge lines into paragraphs.
        # paragraph: True helps structure the output into blocks rather than raw lines.
        # mag_ratio: Magnifies image to detect small text.

        results = reader.readtext(
            processed_bytes,
            paragraph=True,  # Combine lines into paragraphs
            x_ths=0.05,  # Strict horizontal separation (good for columns)
            y_ths=0.6,  # Generous vertical merging
            mag_ratio=1.5  # Slight magnification
        )

        # Results are a list of [[box], text] when paragraph=True
        full_text = '\n\n'.join([res[1] for res in results])

        logging.info(f"EasyOCR detected text length: {len(full_text)}")
        return full_text

    except Exception as e:
        logging.error(f"OCR Error: {e}", exc_info=True)
        return f"ERROR: Failed to process image. {str(e)}"

# --- Gemma Enhancement ---
def enhance_with_gemma(text_to_clean):
    if not text_to_clean or len(text_to_clean.split()) < 2:
        return text_to_clean

    prompt = f"""**Task:** Fix OCR errors in this mixed Arabic/English text.
1. Fix spelling/grammar.
2. Remove random symbols/artifacts.
3. Keep the original meaning and structure.
4. Output ONLY the cleaned text.

**Input:**
{text_to_clean}
"""
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logging.error(f"Gemma Error: {e}")
        return text_to_clean

def process_pdf_hybrid(pdf_bytes):
    """
    Processes a PDF, handling both searchable and scanned-image PDFs.
    1. Tries to extract text directly.
    2. If no text is found, converts pages to images and runs EasyOCR.
    """
    full_text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num, page in enumerate(doc):
            # 1. Try to extract text directly
            page_text = page.get_text()
            
            if not page_text.strip():
                # 2. If no text, it's a scanned image. Run OCR.
                logging.info(f"Page {page_num+1} is a scanned image. Running OCR...")
                
                # Convert page to a high-DPI image
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png") # Convert to PNG bytes
                
                # 3. Run EasyOCR on the image bytes
                page_text = ocr_image_with_easyocr(img_bytes)
                if page_text.startswith("ERROR:"):
                    logging.error(f"Failed to OCR page {page_num+1}: {page_text}")
                    continue # Skip this page on error
            else:
                logging.info(f"Page {page_num+1} is searchable. Extracted text directly.")

            full_text += page_text + "\n\n" # Add newline between pages
            
        doc.close()
        return full_text

    except Exception as e:
        logging.error(f"Failed to process PDF: {e}", exc_info=True)
        return f"ERROR: Failed to process PDF. {str(e)}"

# --- Endpoints ---
@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    raw_text = ocr_image_with_easyocr(file.read())
    if raw_text.startswith("ERROR:"): return jsonify({'error': raw_text}), 500
    return jsonify({'text': enhance_with_gemma(raw_text)})

@app.route('/translate_image_stream', methods=['POST'])
def stream_endpoint():
    data = request.get_data()
    if not data: return jsonify({'error': 'No data'}), 400
    raw_text = ocr_image_with_easyocr(data)
    if raw_text.startswith("ERROR:"): return jsonify({'error': raw_text}), 500
    return jsonify({'text': enhance_with_gemma(raw_text)})

# --- PDF Processing Endpoints (Now with Hybrid Logic) ---
@app.route('/process_pdf', methods=['POST'])
def process_pdf_endpoint():
    """Extracts text from searchable OR scanned-image PDFs."""
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    pdf_bytes = file.read()
    raw_text = process_pdf_hybrid(pdf_bytes)

    if raw_text.startswith("ERROR:"):
        return jsonify({'error': raw_text}), 500

    # Send to Gemma for cleanup
    final_text = enhance_with_gemma(raw_text)
    return jsonify({'text': final_text})
        
@app.route('/process_pdf_stream', methods=['POST'])
def process_pdf_stream_endpoint():
    """Handles a PDF file (searchable or scanned) sent as a raw byte stream."""
    pdf_bytes = request.get_data()
    if not pdf_bytes:
        return jsonify(error="No data received in request body"), 400
        
    raw_text = process_pdf_hybrid(pdf_bytes)

    if raw_text.startswith("ERROR:"):
        return jsonify({'error': raw_text}), 500

    # Send to Gemma for cleanup
    final_text = enhance_with_gemma(raw_text)
    return jsonify({'text': final_text})

# --- Main ---
if __name__ == '__main__':
    logging.info("Starting Offline OCR Server on port 5004...")
    serve(app, host='0.0.0.0', port=5004, threads=20)