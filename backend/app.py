from __future__ import annotations

import base64
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.parse import urlparse

from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image
import requests
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from predict import InvalidImageSourceError, predict_breed


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 MB

app = Flask(
    __name__,
    template_folder=str(BASE_DIR.parent / "templates"),
    static_folder=str(BASE_DIR.parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


DUCKDUCKGO_INSTANT_ANSWER_URL = "https://api.duckduckgo.com/"
DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/?q={}"

BREED_QUERY_ALIASES = {
    "Gir": ["Gyr cattle"],
    "Red Sindhi": ["Red Sindhi cattle", "Sindhi cattle"],
    "Nili Ravi": ["Nili-Ravi buffalo"],
    "Krishna Valley": ["Krishna Valley cattle"],
}


def _is_allowed_filename(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _is_valid_image_file(file_storage: FileStorage) -> bool:
    try:
        with Image.open(file_storage.stream) as image:
            image.verify()
        file_storage.stream.seek(0)
        return True
    except Exception:  # pylint: disable=broad-except
        file_storage.stream.seek(0)
        return False


def _is_valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


@lru_cache(maxsize=256)
def _fetch_duckduckgo_summary(breed_name: str, animal_type: str) -> dict[str, str]:
    query_base = breed_name.replace("_", " ").strip()
    candidates = [
        f"{query_base} {animal_type}",
        f"{query_base} {animal_type} breed",
        f"{query_base} breed",
        query_base,
    ]

    alias_candidates = BREED_QUERY_ALIASES.get(query_base, [])
    candidates = alias_candidates + candidates

    seen = set()
    candidates = [candidate for candidate in candidates if not (candidate in seen or seen.add(candidate))]

    for query in candidates:
        try:
            response = requests.get(
                DUCKDUCKGO_INSTANT_ANSWER_URL,
                timeout=8,
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1,
                    "no_redirect": 1,
                },
                headers={"Accept": "application/json"},
            )
            if response.status_code != 200:
                continue

            payload = response.json()
            abstract_text = (payload.get("AbstractText") or "").strip()
            related_topics = payload.get("RelatedTopics") or []
            source_url = payload.get("AbstractURL", "") or payload.get("Redirect", "")
            source_title = payload.get("Heading", query)

            if not abstract_text:
                topic_queue = list(related_topics)
                while topic_queue and not abstract_text:
                    topic = topic_queue.pop(0)
                    if not isinstance(topic, dict):
                        continue
                    if topic.get("Text"):
                        abstract_text = str(topic["Text"]).strip()
                        source_url = source_url or str(topic.get("FirstURL", ""))
                        source_title = source_title or query_base
                        break
                    nested_topics = topic.get("Topics") or []
                    topic_queue.extend(item for item in nested_topics if isinstance(item, dict))

            if abstract_text:
                return {
                    "summary": abstract_text,
                    "url": source_url,
                    "title": source_title,
                    "source": "DuckDuckGo Instant Answer",
                }
        except Exception:
            continue

    return {
        "summary": "",
        "url": DUCKDUCKGO_SEARCH_URL.format(quote_plus(f"{query_base} {animal_type} breed")),
        "title": query_base,
        "source": "DuckDuckGo Instant Answer",
    }


def _format_prediction(result: dict[str, Any]) -> dict[str, Any]:
    top_prediction = result["top_prediction"]
    top_k = result["top_k"]
    animal_type = top_prediction.get("type", "unknown")
    internet_details = _fetch_duckduckgo_summary(top_prediction["breed"], animal_type)
    fallback_description = top_prediction.get("description", "").strip()
    display_summary = internet_details["summary"] or fallback_description or "Breed details are not available for this prediction."
    source_name = internet_details["source"]
    if not internet_details["summary"] and fallback_description:
        source_name = "Model metadata with DuckDuckGo search link"

    cattle_count = sum(1 for p in top_k if p.get("type") == "cattle")
    buffalo_count = sum(1 for p in top_k if p.get("type") == "buffalo")

    return {
        "source": result["source"],
        "breed": top_prediction["breed"],
        "confidence_percent": round(float(top_prediction["confidence"]) * 100, 2),
        "animal_type": animal_type,
        "description": top_prediction.get("description", ""),
        "internet_details": display_summary,
        "internet_source_url": internet_details["url"],
        "internet_source_title": internet_details["title"],
        "internet_source_name": source_name,
        "top_k": [
            {
                "breed": item["breed"],
                "type": item.get("type", "unknown"),
                "description": item.get("description", ""),
                "confidence_percent": round(float(item["confidence"]) * 100, 2),
            }
            for item in top_k
        ],
        "model_name": result.get("model_name", ""),
        "device": result.get("device", ""),
        "distribution": {
            "cattle": cattle_count,
            "buffalo": buffalo_count,
        },
    }


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    error = ""
    prediction: dict[str, Any] | None = None
    image_source = ""

    if request.method == "POST":
        image_url = request.form.get("image_url", "").strip()
        uploaded_file = request.files.get("image_file")

        # Prefer uploaded image if both are provided.
        if uploaded_file and uploaded_file.filename:
            if not _is_allowed_filename(uploaded_file.filename):
                error = "Unsupported file type. Use JPG, JPEG, PNG, WEBP, or BMP."
            elif not _is_valid_image_file(uploaded_file):
                error = "The uploaded file is not a valid image."
            else:
                temp_path: Path | None = None
                try:
                    suffix = Path(secure_filename(uploaded_file.filename)).suffix.lower() or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        uploaded_file.save(temp_file)
                        temp_path = Path(temp_file.name)

                    result = predict_breed(source=temp_path, top_k=5)
                    prediction = _format_prediction(result)
                    with open(temp_path, "rb") as img_file:
                        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    ext = suffix.lstrip(".").lower()
                    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "bmp": "bmp"}.get(ext, "jpeg")
                    image_source = f"data:image/{mime};base64,{img_b64}"
                except InvalidImageSourceError as exc:
                    error = str(exc)
                except Exception as exc:  # pylint: disable=broad-except
                    error = f"Prediction failed: {exc}"
                finally:
                    if temp_path and temp_path.exists():
                        temp_path.unlink(missing_ok=True)

        elif image_url:
            if not _is_valid_http_url(image_url):
                error = "Please provide a valid image URL starting with http:// or https://"
            else:
                try:
                    result = predict_breed(source=image_url, top_k=5)
                    prediction = _format_prediction(result)
                    image_source = image_url
                except InvalidImageSourceError as exc:
                    error = str(exc)
                except Exception as exc:  # pylint: disable=broad-except
                    error = f"Prediction failed: {exc}"

        else:
            error = "Please upload an image or provide an image URL."

    if request.method == "POST":
        return jsonify({
        "error": error,
        "prediction": prediction,
        "image_source": image_source
    })

    return render_template("index.html",error=error,prediction=prediction,image_source=image_source)



@app.route("/results")
def get_training_results():
    """Serve training results data from the results folder."""
    results_dir = BASE_DIR.parent / "results"
    
    training_results = {
        "confusion_matrix": "/results-image/confusion_matrix.png",
        "datasets": "/results-image/datasets.png",
        "per_class_accuracy": "/results-image/per_class_accuracy.png",
        "training_curves": "/results-image/training_curves.png",
    }
    
    return jsonify(training_results)


@app.route("/results-image/<filename>")
def serve_results_image(filename):
    """Serve images from the results folder."""
    results_dir = BASE_DIR.parent.parent / "results"
    
    # Validate filename to prevent directory traversal
    safe_filename = secure_filename(filename)
    if safe_filename != filename or not filename.endswith(('.png', '.jpg', '.jpeg')):
        return "Invalid filename", 400
    
    file_path = results_dir / safe_filename
    if not file_path.exists():
        return "Image not found", 404
    
    return send_from_directory(str(results_dir), safe_filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
