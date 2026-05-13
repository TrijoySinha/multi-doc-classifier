import "./App.css";
import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    // Dynamic extraction linking to your .env.local string configuration
    const BASE_URL =
      import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

    try {
      // Replaced the hardcoded URL string using template literals
      const response = await fetch(`${BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Server Error");

      const data = await response.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      setResult(data);
    } catch (error) {
      alert("Backend not reachable. Ensure main.py is running.");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const isInvalid = result?.label === "Invalid Document";

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">Document Classifier</h1>
        <p className="subtitle">
          Upload files to verify authenticity using ViT and PaddleOCR pipeline
        </p>

        <div className="upload-section">
          <input
            type="file"
            className="file-input"
            accept="image/*"
            onChange={handleFileChange}
          />

          <button
            className="upload-button"
            onClick={handleUpload}
            disabled={loading || !file}
          >
            {loading ? "Analyzing Layout..." : "Upload & Identify"}
          </button>
        </div>

        {result && (
          <div
            className={`result-container ${isInvalid ? "invalid-layout" : "valid-layout"}`}
          >
            {/* Top Status Header */}
            <div className="status-header">
              <span className="res-tag">Final Status</span>
              <div className="res-value main-label">{result.label}</div>
            </div>

            {/* Structured Meta Grid */}
            <div className="metrics-grid">
              <div className="result-item">
                <span className="res-tag">Decision Source</span>
                <div className="res-value small">{result.source}</div>
              </div>

              <div className="result-item">
                <span className="res-tag">ViT Confidence</span>
                <div className="res-value small">{result.vit_confidence}</div>
              </div>

              <div className="result-item full-width">
                <span className="res-tag">ViT Prediction</span>
                <div className="res-value small">{result.vit_prediction}</div>
              </div>

              <div className="result-item full-width border-top">
                <span className="res-tag">OCR Pipeline Result</span>
                <div className="res-value small">
                  {result.ocr_prediction}{" "}
                  <span className="score-badge">Score: {result.ocr_score}</span>
                </div>
              </div>
            </div>

            {/* Dedicated Text Block */}
            {result.ocr_text_preview && (
              <div className="text-preview-box">
                <span className="res-tag">Extracted Tokens / Match String</span>
                <pre className="preview-text">{result.ocr_text_preview}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
