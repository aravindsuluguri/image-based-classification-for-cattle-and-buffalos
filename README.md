# Cattle and Buffalo Breed Classifier - Quick Start

A simple Flask web app that uses your trained model to classify cattle and buffalo breeds from:

- Uploaded image files (drag and drop or file picker)
- Image URLs (direct image links)

## 1) Requirements

- Python 3.10+
- Internet connection (needed for URL predictions and optional internet details)

## 2) Install

From this folder:

```bash
cd "cattle updated"
pip3 install -r backend/requirements.txt
```

## 3) Run

```bash
python3 app.py
```

Open this in your browser:

- http://127.0.0.1:5001

## 4) How To Use

### Option A: Upload Image

1. Drag and drop an image in the upload area (or click to browse).
2. Click **Classify Image**.

### Option B: Use Image URL

1. Paste a direct image URL (for example, a `.jpg` or `.png` file URL).
2. Click **Classify Image**.

## 5) What You Get

- Predicted breed
- Confidence score
- Animal type (`cattle` or `buffalo`)
- Breed description
- Top-5 predictions
- Internet details summary (when available)

## 6) Common Errors

### "The provided URL does not return an image"

- The URL is likely a web page, not a raw image.
- Use a direct image link.

### "The downloaded content could not be decoded as an image"

- The source may be blocked, invalid, or corrupted.
- Try downloading the image locally and uploading it.

### "Unsupported file type"

Supported upload types are:

- jpg
- jpeg
- png
- webp
- bmp

## 7) Key Files

- `backend/app.py` - Flask web interface
- `app.py` - Root launcher for the backend app
- `backend/predict.py` - Model loading and prediction logic
- `backend/metadata.json` - Class names and breed information
- `backend/best_model.pth` - Trained model weights
- `templates/index.html` - Frontend template
- `static/style.css` - Styling

## 8) Quick Dev Check

```bash
python3 -m py_compile backend/app.py backend/predict.py
```
