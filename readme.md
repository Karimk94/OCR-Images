Hybrid Multilingual OCR & Text Enhancement API

This project is a high-accuracy, multilingual (English & Arabic) OCR and text processing server.

It uses a modern deep-learning approach to extract text from challenging images ("text in the wild") and scanned-image PDFs, and a direct extraction method for searchable PDFs. All extracted text is then programmatically cleaned and enhanced using a local Large Language Model (Gemma) via Ollama.

Key Features

High-Accuracy Image OCR: Uses EasyOCR (PyTorch-based deep learning model) to read text from low-quality images, photos, and complex backgrounds.

Smart Hybrid PDF Processing:

Searchable PDFs: Instantly extracts text using PyMuPDF (fitz).

Scanned PDFs: Automatically converts scanned pages to images and runs them through EasyOCR.

AI-Powered Text Enhancement: All raw extracted text is sent to a local Gemma model (via Ollama) for robust grammar correction, spelling fixes, and removal of OCR artifacts.

Multilingual: Configured out-of-the-box for both English (en) and Arabic (ar).

Portable & Offline Capable: Includes logic to store EasyOCR models within the project directory, allowing the server to run without an internet connection (Ollama must still be running locally).

Ready to Serve: Built as a Flask API served via waitress, a production-ready WSGI server.

Setup and Installation

1. Prerequisites

Python 3.8+

Ollama: This project requires a running Ollama instance to communicate with the Gemma model for text cleanup.

Download and install Ollama for your platform (macOS, Windows, Linux).

Pull the Gemma model:

ollama pull gemma3:12b


Ensure the Ollama server is running.

2. Project Setup

Clone the repository:

git clone <your-repo-url>
cd ocr-py


Create a virtual environment (Recommended):

python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate    # On Windows


Install Python dependencies:

pip install -r requirements.txt


3. Model Configuration

Ollama (Gemma):
Open app.py and verify the Ollama API settings match your local setup (the default is usually correct):

OLLAMA_API_URL = "[http://10.100.61.225:11434/api/generate](http://10.100.61.225:11434/api/generate)"
OLLAMA_MODEL_NAME = "gemma3:12b"


EasyOCR (OCR Models):
The app.py script is configured to store models in a local easyocr_models folder.

The first time you run the server, it will connect to the internet to download the English and Arabic models into this folder. All subsequent runs will be fast and can work offline.

Running the Server

Once all dependencies and models are in place, start the production server:

python3 app.py


The server will start on http://0.0.0.0:5004 and will log the EasyOCR model loading process.

API Endpoints

Image Processing

These endpoints are for processing image files (.png, .jpg, etc.).

/ocr

Method: POST

Form-Data: file (the image file)

Description: Extracts text from a single image.

/translate_image

Method: POST

Form-Data: image_file (the image file)

Description: Identical to /ocr, provided for legacy compatibility.

/translate_image_stream

Method: POST

Body: Raw image bytes (e.g., sent via curl --data-binary "@my_image.png")

Description: Extracts text from a raw image byte stream.

PDF Processing

These "smart" endpoints handle both searchable and scanned-image PDFs.

/process_pdf

Method: POST

Form-Data: pdf_file (the PDF file)

Description: Extracts text from all pages of a PDF.

/process_pdf_stream

Method: POST

Body: Raw PDF bytes

Description: Extracts text from a raw PDF byte stream.

Example curl Usage

Image File:

curl -X POST -F "file=@/path/to/my_image.png" [http://127.0.0.1:5004/ocr](http://127.0.0.1:5004/ocr)


PDF File:

curl -X POST -F "pdf_file=@/path/to/my_document.pdf" [http://127.0.0.1:5004/process_pdf](http://127.0.0.1:5004/process_pdf)


Image Stream:

curl -X POST --data-binary "@/path/to/my_image.png" [http://127.0.0.1:5004/translate_image_stream](http://127.0.0.1:5004/translate_image_stream)
