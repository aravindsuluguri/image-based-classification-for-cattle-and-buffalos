import { html, useRef, useState, useEffect } from "./react-lib.js";

function capitalize(v) { return v ? v.charAt(0).toUpperCase() + v.slice(1) : ""; }

function ConfidenceBar({ percent }) {
  const color = percent >= 70 ? "#00b894" : percent >= 40 ? "#fdcb6e" : "#ff7675";
  return html`<div className="confidence-bar-wrap">
    <div className="confidence-bar-track"><div className="confidence-bar-fill" style=${{ width: percent + "%", background: color }} /></div>
    <span className="confidence-bar-label" style=${{ color }}>${percent}%</span>
  </div>`;
}

/* ===== Camera Modal ===== */
export function CameraModal({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState("");
  const [countdown, setCountdown] = useState(null);

  useEffect(() => {
    let s = null;
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 960 } } })
      .then((mediaStream) => {
        s = mediaStream;
        setStream(mediaStream);
        if (videoRef.current) { videoRef.current.srcObject = mediaStream; videoRef.current.play(); }
      })
      .catch(() => setError("Could not access camera. Please allow camera permission and try again."));
    return () => { if (s) s.getTracks().forEach(t => t.stop()); };
  }, []);

  const capture = () => {
    setCountdown(3);
    let c = 3;
    const timer = setInterval(() => {
      c--;
      if (c <= 0) {
        clearInterval(timer);
        setCountdown(null);
        doCapture();
      } else {
        setCountdown(c);
      }
    }, 600);
  };

  const doCapture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        if (stream) stream.getTracks().forEach(t => t.stop());
        onCapture(file);
      }
    }, "image/jpeg", 0.92);
  };

  return html`
    <div className="camera-overlay" onClick=${onClose}>
      <div className="camera-modal" onClick=${(e) => e.stopPropagation()}>
        <div className="camera-header">
          <h3>📸 Camera Capture</h3>
          <button className="camera-close" onClick=${onClose}>✕</button>
        </div>
        ${error ? html`<div className="camera-error">${error}</div>` : html`
          <div className="camera-viewport">
            <video ref=${videoRef} autoplay playsinline muted className="camera-video" />
            ${countdown !== null ? html`<div className="camera-countdown">${countdown}</div>` : null}
          </div>
          <canvas ref=${canvasRef} style=${{ display: "none" }} />
          <div className="camera-actions">
            <button className="camera-capture-btn" onClick=${capture}>
              <span className="capture-ring"></span>
            </button>
            <p className="camera-hint">Point at the animal and tap to capture</p>
          </div>
        `}
      </div>
    </div>
  `;
}

/* ===== Loading Spinner ===== */
function LoadingOverlay() {
  return html`<div className="loading-overlay">
    <div className="loading-spinner"></div>
    <p className="loading-text">Analyzing image with AI...</p>
    <p className="loading-sub">Running 4x augmented predictions</p>
  </div>`;
}

/* ===== Dashboard ===== */
export function DashboardPanel({ prediction, totalScans, recentItems, onTrainingResults }) {
  return html`
    <section className="dashboard card animate-in">
      <header className="dash-header">
        <p className="brand">BreedLens</p>
        <nav>
          <a href="#" onClick=${(e) => { e.preventDefault(); onTrainingResults?.(); }}>Training Results</a>
          <a href="#">History</a>
          <a href="#">Encyclopedia</a>
          <span className="avatar">ID</span>
        </nav>
      </header>
      <article className="hero-banner"><div className="hero-copy">
        <h1>Image Based Classification on Animal Type: Cattle and Buffalo</h1>
        <p>AI-powered cattle and buffalo recognition for veterinarians, farms, and research teams.</p>
        <a className="cta" href="#identify">Start New Classification</a>
      </div></article>
      <section className="stats-row">
        <div className="stat-card"><p className="label">Total Scans</p><p className="value">${totalScans}</p></div>
        <div className="stat-card"><p className="label">Animals Known</p><p className="value">36</p></div>
        <div className="stat-card"><p className="label">Top-5 Accuracy</p><p className="value">~82%</p></div>
      </section>
      <section className="dash-content">
        <div className="recent-list">
          <div className="block-head"><h2>Recent Classifications</h2><a href="#">View all</a></div>
          <ul>${recentItems.map((item) => html`
            <li key=${item.title}><div><p className="item-title">${item.title}</p><p className="item-sub">${item.subtitle}</p></div>
            <span className=${`pill ${item.type}`}>${capitalize(item.type)}</span></li>`)}</ul>
        </div>
        <article className="featured-breed">
          <p className="featured-tag">Featured Animal</p><div className="feature-image"></div>
          <h3>${prediction?.breed || "Sahiwal Cattle"}</h3>
          <p>${prediction?.description || "Native to South Asia, known for heat tolerance and strong dairy productivity in tropical climates."}</p>
          ${prediction?.internet_source_url ? html`<a href=${prediction.internet_source_url} target="_blank" rel="noopener noreferrer">Learn more</a>` : html`<a href="#">Learn more</a>`}
        </article>
      </section>
    </section>`;
}

/* ===== Results Page ===== */
export function ResultsPage({ imageSource, prediction }) {
  if (!prediction) return null;
  return html`
    <section className="results-page animate-in">
      <header className="results-header card">
        <div>
          <p className="results-kicker">Animal Identification Result</p>
          <h1>${prediction.breed}</h1>
          <p className="results-subtitle">Predicted as ${capitalize(prediction.animal_type)} with ${prediction.confidence_percent}% confidence.</p>
        </div>
        <a className="results-back" href="/">Analyze another image</a>
      </header>
      <section className="results-layout">
        <article className="result-hero card">
          ${imageSource ? html`<div className="result-image-frame"><img src=${imageSource} alt="Submitted animal" loading="lazy" /></div>` : null}
          <div className="result-overview">
            <p className=${`pill ${prediction.animal_type}`}>${capitalize(prediction.animal_type)}</p>
            <h2>${prediction.breed}</h2>
            <p className="result-description">${prediction.description || "No local breed description available."}</p>
            <div className="result-confidence-section">
              <p className="result-confidence-label">Confidence</p>
              <${ConfidenceBar} percent=${prediction.confidence_percent} />
            </div>
            <div className="result-stats">
              <div><span>Predicted type</span><strong>${capitalize(prediction.animal_type)}</strong></div>
              <div><span>Cattle votes</span><strong>${prediction.distribution.cattle}</strong></div>
              <div><span>Buffalo votes</span><strong>${prediction.distribution.buffalo}</strong></div>
            </div>
            <p className="result-meta">Model ${prediction.model_name} · Device ${prediction.device} · TTA 4x</p>
          </div>
        </article>
        <article className="internet-panel card">
          <p className="panel-label">Internet Details</p>
          <h3>${prediction.internet_source_title || prediction.breed}</h3>
          <p>${prediction.internet_details}</p>
          <p className="source-chip">Source: ${prediction.internet_source_name || "DuckDuckGo Instant Answer"}</p>
          ${prediction.internet_source_url ? html`<a className="detail-link" href=${prediction.internet_source_url} target="_blank" rel="noopener noreferrer">Open source reference</a>` : null}
        </article>
        <article className="top-predictions card">
          <div className="block-head"><h2>Top Predictions</h2></div>
          <ul>${prediction.top_k.map((item) => html`
            <li key=${item.breed} className="top-pred-item">
              <div className="top-pred-info"><p className="item-title">${item.breed}</p><p className="item-sub">${item.description || "Animal detail not available."}</p></div>
              <div className="prediction-side"><span className=${`pill ${item.type}`}>${capitalize(item.type)}</span><${ConfidenceBar} percent=${item.confidence_percent} /></div>
            </li>`)}</ul>
        </article>
      </section>
    </section>`;
}

/* ===== Identify Panel ===== */
export function IdentifyPanel({ chooseFile, clearFile, error, fileInputRef, formRef, imageSource, imageUrl, isDragging, isLoading, onDrop, onSubmit, prediction, previewUrl, selectedFile, setImageUrl, setIsDragging, onOpenCamera }) {
  return html`
    <section className="identify card animate-in" id="identify">
      <header className="identify-header"><p className="brand">BreedAI</p><div className="header-actions"><a href="#">History</a></div></header>
      <div className="identify-layout">
        <section className="upload-column">
          <h2>Animal Identification</h2>
          <p className="lead">Upload a photo, use your camera, or paste an image URL to identify the breed instantly.</p>
          <form ref=${formRef} method="post" encType="multipart/form-data" action="/" onSubmit=${onSubmit}>
            <div className=${`dropzone ${isDragging ? "dragging" : ""} ${previewUrl ? "has-preview" : ""}`}
              role="button" tabIndex="0" aria-label="Drag and drop image upload area"
              onClick=${() => !previewUrl && fileInputRef.current?.click()}
              onDragEnter=${(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }}
              onDragOver=${(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }}
              onDragLeave=${(e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); }}
              onDrop=${onDrop}>
              ${previewUrl ? html`
                <div className="preview-container">
                  <img src=${previewUrl} alt="Preview" className="preview-image" />
                  <div className="preview-overlay">
                    <p className="preview-filename">${selectedFile?.name || "Image selected"}</p>
                    <button type="button" className="preview-change-btn" onClick=${(e) => { e.stopPropagation(); clearFile(); }}>✕ Remove</button>
                  </div>
                </div>
              ` : html`
                <p className="upload-icon" aria-hidden="true">⇪</p>
                <p className="drop-title">Click to upload or drag and drop</p>
                <p className="drop-sub">PNG, JPG or WEBP (max 10MB)</p>
              `}
              <input ref=${fileInputRef} name="image_file" type="file" accept="image/*" onChange=${(e) => chooseFile(e.target.files?.[0])} />
            </div>
            <div className="row-actions">
              <button type="button" className="secondary" onClick=${onOpenCamera}>📷 Open Camera</button>
              <button type="submit" className="primary">${isLoading ? "⏳ Analyzing..." : "🔍 Classify Image"}</button>
            </div>
            <label htmlFor="image-url">or paste an image URL</label>
            <input id="image-url" name="image_url" type="text" value=${imageUrl}
              onInput=${(e) => setImageUrl(e.target.value)} placeholder="https://example.com/cattle.jpg" />
          </form>
          ${error ? html`<div className="message error animate-shake">${error}</div>` : null}
          <p className="privacy-note">🔒 Your privacy matters. Images are processed only for prediction and never shared.</p>
        </section>
        <${TipsColumn} />
      </div>
    </section>`;
}

function TipsColumn() {
  return html`<aside className="tips-column">
    <h3>Pro Tips for Better Results</h3>
    <ul>
      <li><span>1</span>Use good natural lighting.</li>
      <li><span>2</span>Capture a full side profile.</li>
      <li><span>3</span>Focus on patterns and hump shape.</li>
      <li><span>4</span>Avoid motion blur and low contrast.</li>
    </ul>
    <p className="mini-note">Supports 36 cattle and buffalo breeds.</p>
  </aside>`;
}

/* ===== Training Results (unchanged) ===== */
export function TrainingResultsPage() {
  const trainingResults = [
    { id: "training_curves", title: "Training Curves", description: "Model loss and accuracy over training epochs", image: "/results-image/training_curves.png" },
    { id: "confusion_matrix", title: "Confusion Matrix", description: "Prediction accuracy breakdown by animal class", image: "/results-image/confusion_matrix.png" },
    { id: "per_class_accuracy", title: "Per-Class Accuracy", description: "Individual animal recognition accuracy metrics", image: "/results-image/per_class_accuracy.png" },
    { id: "datasets", title: "Dataset Overview", description: "Training and validation data distribution", image: "/results-image/datasets.png" },
  ];
  const modelBenchmarks = ["CNN", "ANN", "SVM", "KNN", "Decision Tree"];
  const performanceMetrics = [
    { label: "Precision", value: "61.3%" }, { label: "Accuracy", value: "61.0%" },
    { label: "Recall", value: "61.0%" }, { label: "F1 Score", value: "60.9%" },
    { label: "Aug Score", value: "82.3%" }, { label: "G Mean Score", value: "54.9%" },
  ];
  const trainingConfigRows = [
    ["Model Architecture", "EfficientNetV2-M (tf_efficientnetv2_m)"], ["Backbone", "EfficientNetV2-M via timm (ImageNet-21k pretrained)"],
    ["Classification Head", "BN -> Dropout(0.35) -> Linear(in, 512) -> ReLU -> BN -> Dropout(0.175) -> Linear(512, 36)"],
    ["Image Size", "384 x 384 px"], ["Batch Size", "16"], ["Epochs", "40 (with early stopping)"],
    ["Optimizer", "AdamW"], ["Base Learning Rate", "3 x 10^-4"],
    ["LR Schedule", "Cosine Annealing with Linear Warmup (3 epochs)"], ["Min LR", "1 x 10^-6"],
    ["Weight Decay", "1 x 10^-4"], ["Loss Function", "Label Smoothing Cross-Entropy (epsilon = 0.1)"],
    ["Regularisation", "MixUp (alpha = 0.4), CoarseDropout, Dropout (0.35)"], ["Mixed Precision", "AMP (FP16 via GradScaler)"],
    ["Target Images/Class", "~1,000"], ["Train / Val / Test Split", "75% / 15% / 10%"],
  ];
  const perClassRows = [
    ["Kankrej","Cattle","0.89"],["Sahiwal","Cattle","0.88"],["Gir","Cattle","0.86"],["Tharparkar","Cattle","0.82"],
    ["Kangayam","Cattle","0.77"],["Pandharpuri","Buffalo","0.71"],["Khillari","Cattle","0.69"],["Jaffarabadi","Buffalo","0.68"],
    ["Murrah","Buffalo","0.67"],["Banni","Buffalo","0.66"],["Nili_Ravi","Buffalo","0.66"],["Punganur","Cattle","0.65"],
    ["Rathi","Cattle","0.65"],["Ongole","Cattle","0.63"],["Deoni","Cattle","0.63"],["Hallikar","Cattle","0.63"],
    ["Bargur","Cattle","0.58"],["Vechur","Cattle","0.57"],["Krishna_Valley","Cattle","0.53"],["Red_Sindhi","Cattle","0.50"],
    ["Umblachery","Cattle","0.50"],["Dangi","Cattle","0.48"],["Mehsana","Buffalo","0.48"],["Toda","Buffalo","0.44"],
    ["Malvi","Cattle","0.40"],["Amritmahal","Cattle","0.40"],["Tarai","Cattle","0.39"],["Gaolao","Cattle","0.39"],
    ["Chilika","Buffalo","0.39"],["Nagpuri","Buffalo","0.39"],["Nagori","Cattle","0.38"],["Surti","Buffalo","0.35"],
    ["Hariana","Cattle","0.34"],["Bhadawari","Buffalo","0.26"],["Bachaur","Cattle","0.22"],["Siri","Cattle","0.15"],
  ];
  const comparisonRows = [
    ["Proposed (EfficientNetV2-M)","Breed Classification","36","~63% (macro avg)","Val Top-5: ~82%; new 36-class benchmark"],
    ["Arya et al. (2025)","Breed Classification","-","72.5%","Lightweight CNN, fewer breeds"],
    ["Warhade et al. (2023)","Breed ID (Sahiwal/Red Sindhi)","2","82%","EfficientNet-B0, 2-class only"],
    ["Gupta et al. (2021)","Breed Detection","8","81.07%","YOLOv4, 8 dairy breeds"],
    ["Pan et al. (2022)","Buffalo Breed Classification","~6","95.90%","SA-CNN, fewer classes"],
    ["MHAFF (2025)","Cattle Identification","-","99.88%","Individual ID, not breed-level"],
  ];
  return html`
    <section className="training-results-page">
      <header className="training-header card"><div>
        <p className="results-kicker">Model Training</p><h1>Training Results</h1>
        <p className="results-subtitle">Comprehensive analysis of the BreedLens AI model performance and training metrics.</p>
      </div><a className="training-back" href="/">Back to Dashboard</a></header>
      <section className="results-grid">${trainingResults.map((r) => html`
        <article className="results-card card" key=${r.id}><div className="results-image-container"><img src=${r.image} alt=${r.title} loading="lazy" /></div>
        <div className="results-card-content"><h3>${r.title}</h3><p className="results-card-label">${r.description}</p></div></article>`)}</section>
      <section className="ml-results-grid">
        <article className="ml-block card"><p className="panel-label">Machine Learning Models</p><h3>Model Benchmarks</h3>
          <ul className="model-list">${modelBenchmarks.map((m) => html`<li key=${m}><span>${m}</span></li>`)}</ul></article>
        <article className="ml-block card"><p className="panel-label">Performance Metrics</p><h3>Evaluation Summary</h3>
          <div className="metric-grid">${performanceMetrics.map((m) => html`<div className="metric-item" key=${m.label}><span>${m.label}</span><strong>${m.value}</strong></div>`)}</div></article>
      </section>
      <section className="research-report card">
        <h2>Methodology and Results Write-up</h2>
        <article className="research-section"><h3>A. Problem Formulation</h3><p>The task is formulated as a supervised 36-class image classification problem. Given an input image x in R^(H x W x C), the model predicts a class label y-hat in {0, 1, ..., 35} corresponding to one of 36 Indian bovine breeds. The 36 classes comprise 23 cattle breeds and 13 buffalo breeds.</p></article>
        <article className="research-section"><h3>B. Dataset Construction</h3><p>Three Kaggle datasets were combined: Breed_cattle_buffalo, Indian Cattle and Buffalo Breeds, and Indian Bovine breeds. Images were mapped to canonical breed names. Additional images were scraped from Bing. All images were validated for minimum resolution (224 x 224 px). Near-duplicate images were removed using perceptual hashing (pHash threshold less than 8). A target of approximately 1,000 images per class was set. The final dataset was split into training (75%), validation (15%), and test (10%) subsets.</p></article>
        <article className="research-section"><h3>C. Model Architecture</h3><p>The BreedClassifier model uses EfficientNetV2-M as its backbone with ImageNet-21k pretrained weights. A custom classification head is attached: BatchNorm1d -> Dropout(0.35) -> Linear(in_features, 512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.175) -> Linear(512, 36). Training proceeded in three stages: (1) head-only, (2) partial fine-tuning, (3) full model fine-tuning with differential learning rates.</p></article>
        <article className="research-section"><h3>D. Training Configuration</h3><p>Table 1. Training Hyperparameters and Configuration</p>
          <div className="research-table-wrap"><table className="research-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>${trainingConfigRows.map(([p,v]) => html`<tr key=${p}><td>${p}</td><td>${v}</td></tr>`)}</tbody></table></div></article>
        <article className="research-section"><h3>E. Results</h3>
          <h4>Per-Class Accuracy</h4><p>Table 2. Per-Class Accuracy for All 36 Breeds</p>
          <div className="research-table-wrap"><table className="research-table"><thead><tr><th>Breed</th><th>Type</th><th>Accuracy</th></tr></thead>
          <tbody>${perClassRows.map(([b,t,a]) => html`<tr key=${b}><td>${b}</td><td>${t}</td><td>${a}</td></tr>`)}</tbody></table></div>
          <h4>Comparison with Prior Work</h4><p>Table 3. Comparison with Related Work</p>
          <div className="research-table-wrap"><table className="research-table"><thead><tr><th>Model / Study</th><th>Task</th><th># Classes</th><th>Top-1 Accuracy</th><th>Notes</th></tr></thead>
          <tbody>${comparisonRows.map(([m,t,c,a,n]) => html`<tr key=${m}><td>${m}</td><td>${t}</td><td>${c}</td><td>${a}</td><td>${n}</td></tr>`)}</tbody></table></div>
          <p className="research-note">Higher accuracies reported in prior works often use fewer classes. This 36-class setting is substantially harder (random chance = 2.8%), and Top-5 around 82% is practically useful for expert-assisted workflows.</p>
        </article>
      </section>
    </section>`;
}
