from flask import Flask, request, jsonify
from PIL import Image
import fitz
import logging
from flask_cors import CORS
from waitress import serve
import io
import requests
import json
import os
import easyocr

# --- Basic Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Configuration ---
# OLLAMA API Endpoint
OLLAMA_API_URL = "http://10.100.61.225:11434/api/generate"
OLLAMA_MODEL_NAME = "gemma3:12b"

# --- Model Initialization ---
# Initialize the EasyOCR reader once when the server starts.

# Define a path for models *inside* our project directory
model_dir = os.path.join(os.path.dirname(__file__), 'easyocr_models')
os.makedirs(model_dir, exist_ok=True) # Ensure the directory exists

try:
    logging.info(f"Loading EasyOCR models from: {model_dir}")
    reader = easyocr.Reader(['en', 'ar'], gpu=True, model_storage_directory=model_dir)
    logging.info("EasyOCR models loaded successfully.")
except Exception as e:
    logging.error(f"Failed to initialize EasyOCR: {e}", exc_info=True)
    reader = None

# --- Gemma Enhancement Function ---

def enhance_with_gemma(text_to_clean):
    """
    Sends the cleaned OCR text to a local Gemma model via Ollama for enhancement.
    """
    if not text_to_clean or len(text_to_clean.split()) < 2:
        logging.info("Text is too short for Gemma enhancement, skipping.")
        return text_to_clean

    prompt = f"""**Role:** You are an expert multilingual editor specializing in correcting and refining text extracted by an Optical Character Recognition (OCR) tool.

**Context:** The following text was extracted from a document (image or PDF) that contains a mix of English and Arabic. The extraction process may have introduced errors.

**Task:** Your objective is to meticulously clean and correct the provided text. Follow these instructions precisely:
1.  **Correct Spelling and Grammar:** Fix all spelling and grammatical errors in both the English and Arabic portions of the text.
2.  **Remove Artifacts:** Eliminate any characters or symbols that are clearly OCR errors.
3.  **Ensure Coherence:** Re-structure the text into logical sentences and paragraphs, while strictly preserving the original meaning.
4.  **Final Output:** Provide ONLY the fully cleaned and corrected text. Do not include any introductory phrases.

**Raw OCR Text to Clean:**
```
{text_to_clean}
```"""

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        logging.info(f"Sending text to Gemma ({OLLAMA_MODEL_NAME}) for enhancement: '{text_to_clean[:70]}...'")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
        response.raise_for_status()
        
        response_data = response.json()
        enhanced_text = response_data.get("response", "").strip()

        logging.info(f"Received enhanced text from Gemma: '{enhanced_text[:70]}...'")
        return enhanced_text if enhanced_text else text_to_clean

    except requests.exceptions.ConnectionError:
        logging.error(f"Could not connect to Ollama API at {OLLAMA_API_URL}.")
        return text_to_clean
    except Exception as e:
        logging.error(f"An unexpected error occurred when calling Gemma: {e}", exc_info=True)
        return text_to_clean

# --- OCR Core Logic Functions ---

def ocr_image_with_easyocr(image_stream_bytes):
    """
    Performs OCR on image bytes using EasyOCR.
    """
    if reader is None:
        logging.error("EasyOCR reader is not initialized.")
        return "ERROR: OCR Service is not ready."

    try:
        # reader.readtext() accepts image bytes directly
        result = reader.readtext(image_stream_bytes)
        
        # Join all the detected text blocks with a newline.
        full_text = '\n'.join([res[1] for res in result])
        
        logging.info(f"EasyOCR raw output: '{full_text[:200]}...'")
        return full_text

    except Exception as e:
        logging.error(f"An error occurred during EasyOCR processing: {e}", exc_info=True)
        return f"ERROR: Failed to process image for OCR. {str(e)}"

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

# --- API Endpoints ---

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    """Generic OCR endpoint for processing image files."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_stream = file.read()
    raw_ocr_text = ocr_image_with_easyocr(image_stream)
    
    if raw_ocr_text.startswith("ERROR:"):
        return jsonify({'error': raw_ocr_text}), 500
    
    # Send to Gemma for cleanup
    final_text = enhance_with_gemma(raw_ocr_text)
    return jsonify({'text': final_text})

@app.route('/translate_image', methods=['POST'])
def translate_image_endpoint():
    """Endpoint for image translation with post-processing."""
    if 'image_file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_stream = file.read()
    raw_ocr_text = ocr_image_with_easyocr(image_stream)
    
    if raw_ocr_text.startswith("ERROR:"):
        return jsonify({'error': raw_ocr_text}), 500
    
    # Send to Gemma for cleanup
    final_text = enhance_with_gemma(raw_ocr_text)
    return jsonify({'text': final_text})

@app.route('/translate_image_stream', methods=['POST'])
def translate_image_stream_endpoint():
    """Handles a single image sent as a raw byte stream with post-processing."""
    image_bytes = request.get_data()
    if not image_bytes:
        return jsonify(error="No data received in request body"), 400

    raw_ocr_text = ocr_image_with_easyocr(image_bytes)

    if raw_ocr_text.startswith("ERROR:"):
        return jsonify({'error': raw_ocr_text}), 500
    
    # Send to Gemma for cleanup
    final_text = enhance_with_gemma(raw_ocr_text)
    return jsonify({'text': final_text})


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

# --- Main Execution ---
if __name__ == '__main__':
    logging.info("Starting Flask server with Waitress...")
    serve(app, host='0.0.0.0', port=5004, threads=100)