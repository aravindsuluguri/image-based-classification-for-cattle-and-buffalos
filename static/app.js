import { createRoot, html, useRef, useState, useEffect } from "./react-lib.js";
import { DashboardPanel, IdentifyPanel, ResultsPage, TrainingResultsPage, CameraModal } from "./components.js";

function App() {
  const initial = window.__INITIAL_STATE__ || {};
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction,setPrediction] = useState(initial.prediction || null);
  const [imageSource,setImageSource] = useState(initial.image_source || "");
  const [error,setError] = useState(initial.error || "");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [currentPage, setCurrentPage] = useState(() => {
    if (window.location.hash === "#training-results") return "training-results";
    return prediction ? "results" : "dashboard";
  });

  const fileInputRef = useRef(null);
  const formRef = useRef(null);

  useEffect(() => {
    const handleHashChange = () => {
      if (window.location.hash === "#training-results") setCurrentPage("training-results");
      else if (window.location.hash === "") setCurrentPage("dashboard");
    };
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const totalScans = prediction ? 128 : 127;
  const recentItems = prediction
    ? prediction.top_k.slice(0, 2).map((item) => ({
        title: item.breed, subtitle: `${item.type} classification`, type: item.type,
      }))
    : [
        { title: "Murrah Buffalo", subtitle: "Buffalo classification · 2h ago", type: "buffalo" },
        { title: "Gir Cattle", subtitle: "Cattle classification · Yesterday", type: "cattle" },
      ];

  const chooseFile = (file) => {
    if (!file) return;
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(file);
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreviewUrl("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      if (fileInputRef.current) {
        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInputRef.current.files = transfer.files;
      }
      chooseFile(file);
    }
  };

  const handleCameraCapture = (file) => {
    setShowCamera(false);
    if (fileInputRef.current) {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      fileInputRef.current.files = transfer.files;
    }
    chooseFile(file);
  };

 const handleSubmit = async (e) => {
    e.preventDefault();

    setIsLoading(true);
    setError("");

  try {
    const formData = new FormData();
    if (selectedFile) {
      formData.append("image_file", selectedFile);
    }

    if (imageUrl) {
      formData.append("image_url", imageUrl);
    }

    const response = await fetch("/", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      setError(data.error);
    } else {
      setPrediction(data.prediction);
      setImageSource(data.image_source);
      setCurrentPage("results");
    }

    } catch (err) {
      setError("Failed to classify image");
    } finally {
      setIsLoading(false);
    }
  };


  if (currentPage === "training-results") {
    return html`<main className="shell"><${TrainingResultsPage} /></main>`;
  }

  if (prediction && currentPage === "results") {
    return html`<main className="shell"><${ResultsPage} imageSource=${imageSource} prediction=${prediction} /></main>`;
  }

  return html`
    <main className="shell">
      <${DashboardPanel} prediction=${prediction} totalScans=${totalScans} recentItems=${recentItems}
        onTrainingResults=${() => { window.location.hash = "#training-results"; setCurrentPage("training-results"); }} />
      <${IdentifyPanel}
        chooseFile=${chooseFile} clearFile=${clearFile} error=${error} fileInputRef=${fileInputRef} formRef=${formRef}
        imageSource=${imageSource} imageUrl=${imageUrl} isDragging=${isDragging} isLoading=${isLoading}
        onDrop=${onDrop} onSubmit=${handleSubmit} prediction=${prediction} previewUrl=${previewUrl}
        selectedFile=${selectedFile} setImageUrl=${setImageUrl} setIsDragging=${setIsDragging}
        onOpenCamera=${() => setShowCamera(true)} />
      ${showCamera ? html`<${CameraModal} onCapture=${handleCameraCapture} onClose=${() => setShowCamera(false)} />` : null}
    </main>
  `;
}

const container = document.getElementById("root");
if (container) createRoot(container).render(html`<${App} />`);
