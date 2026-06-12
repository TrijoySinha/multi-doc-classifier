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

    const BASE_URL =
      import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

    try {
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

  const isWarning = Boolean(result?.warning);

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">Document Classifier</h1>

        <p className="subtitle">
          Upload a document image to classify it using a vision model with OCR-assisted verification.
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
            {loading ? "Analyzing..." : "Upload & Identify"}
          </button>
        </div>

        {result && (
          <div className={`result-container ${isWarning ? "invalid-layout" : "valid-layout"}`}>
            <div className="status-header">
              <span className="res-tag">Final Prediction</span>
              <div className="res-value main-label">{result.label}</div>
            </div>

            {result.possible_actual_document && (
              <div className="result-item full-width">
                <span className="res-tag">Possible Actual Document</span>
                <div className="res-value">{result.possible_actual_document}</div>
              </div>
            )}

            <div className="metrics-grid">
              <div className="result-item full-width">
                <span className="res-tag">Model Prediction</span>
                <div className="res-value">{result.raw_model_label}</div>
              </div>

              <div className="result-item full-width">
                <span className="res-tag">Confidence</span>
                <div className="res-value">
                  {result.confidence_percent}
                </div>
              </div>

              <div className="result-item full-width">
                <span className="res-tag">Prediction Margin</span>
                <div className="res-value">{result.margin}</div>
              </div>

              {result.ocr_best_known_class && (
                <div className="result-item full-width">
                  <span className="res-tag">OCR Suggested Known Class</span>
                  <div className="res-value">{result.ocr_best_known_class}</div>
                </div>
              )}
            </div>

            {result.warning && (
              <div className="result-item full-width">
                <span className="res-tag">Warning</span>
                <div className="res-value">{result.warning}</div>
              </div>
            )}

            {result.top_predictions && (
              <div className="result-item full-width">
                <span className="res-tag">Top Predictions</span>
                <div className="res-value">
                  {result.top_predictions.map((item, index) => (
                    <div key={index}>
                      {item.label}: {(item.confidence * 100).toFixed(2)}%
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.ocr_text_preview && (
              <div className="result-item full-width">
                <span className="res-tag">OCR Text Preview</span>
                <div className="res-value">{result.ocr_text_preview}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;