import { useState } from 'react';
import './App.css';

const API_BASE = 'http://localhost:8000';

const GRADE_COLORS = {
  0: '#22c55e',   // green
  1: '#84cc16',   // lime
  2: '#f59e0b',   // amber
  3: '#f97316',   // orange
  4: '#ef4444',   // red
};

const GRADE_LABELS = {
  0: 'No apparent DR',
  1: 'Mild NPDR',
  2: 'Moderate NPDR',
  3: 'Severe NPDR',
  4: 'Proliferative DR',
};

function ConfidenceBar({ label, value, grade }) {
  const color = GRADE_COLORS[grade] ?? '#60a5fa';
  return (
    <div style={{ marginBottom: '0.6rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
        <span>{label}</span>
        <span>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${(value * 100).toFixed(1)}%`, background: color, borderRadius: '4px', transition: 'width 0.8s ease' }} />
      </div>
    </div>
  );
}

function ConfidenceTierBadge({ tier }) {
  const tierConfig = {
    HIGH: { color: '#22c55e', bg: 'rgba(34,197,94,0.15)', label: '● HIGH CONFIDENCE' },
    MODERATE: { color: '#f59e0b', bg: 'rgba(245,158,11,0.15)', label: '● MODERATE CONFIDENCE' },
    LOW: { color: '#ef4444', bg: 'rgba(239,68,68,0.15)', label: '● LOW CONFIDENCE' },
  };
  const cfg = tierConfig[tier] ?? tierConfig.LOW;
  return (
    <span style={{
      fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.05em',
      color: cfg.color, background: cfg.bg,
      border: `1px solid ${cfg.color}44`,
      borderRadius: '4px', padding: '2px 8px'
    }}>
      {cfg.label}
    </span>
  );
}

function ImagePanel({ title, src, fallback }) {
  const [error, setError] = useState(false);
  return (
    <div className="glass-panel result-card">
      <h3>{title}</h3>
      {src && !error
        ? <img src={src} alt={title} onError={() => setError(true)} />
        : <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'160px', color:'var(--text-muted)', fontSize:'0.85rem', background:'rgba(255,255,255,0.03)', borderRadius:'8px' }}>
            {fallback}
          </div>
      }
    </div>
  );
}

function App() {
  const [fileObj, setFileObj] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [reviewSubmitted, setReviewSubmitted] = useState(false);
  const [reviewLoading, setReviewLoading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFileObj(e.target.files[0]);
      setPreviewUrl(URL.createObjectURL(e.target.files[0]));
      setResults(null);
      setError(null);
      setReviewSubmitted(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
      setFileObj(dropped);
      setPreviewUrl(URL.createObjectURL(dropped));
      setResults(null);
      setError(null);
      setReviewSubmitted(false);
    }
  };

  const analyzeImage = async () => {
    if (!fileObj) return;
    setAnalyzing(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', fileObj);
      const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || errData.error || `Server error ${response.status}`);
      }
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const submitReview = async (decision) => {
    if (!results?.image_id) return;
    setReviewLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_id: results.image_id, decision, comment: '' }),
      });
      if (resp.ok) setReviewSubmitted(true);
    } catch (e) {
      console.error('Review submission failed:', e);
    } finally {
      setReviewLoading(false);
    }
  };

  const reset = () => {
    setFileObj(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    setReviewSubmitted(false);
  };

  const grade = results?.prediction?.grade;
  const gradeColor = grade !== undefined ? GRADE_COLORS[grade] : '#60a5fa';
  const tier = results?.prediction?.confidence_tier;

  return (
    <div className="dashboard">
      <header className="header">
        <h1>DR-Screening-XAI Portal</h1>
        <p style={{ color: 'var(--text-muted)' }}>
          Upload a fundus image for explainable diabetic retinopathy screening
        </p>
      </header>

      {/* ─── Upload Panel ─── */}
      {!results && !analyzing && (
        <div className="glass-panel" style={{ maxWidth: '600px', margin: '0 auto' }}>
          <label
            className="upload-area"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <svg width="48" height="48" fill="none" stroke="var(--text-muted)" viewBox="0 0 24 24" style={{ marginBottom: '1rem' }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            <h3>Click or drag image to upload</h3>
            <p style={{ color: 'var(--text-muted)' }}>Supports JPG, PNG (High resolution recommended)</p>
            <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
          </label>

          {previewUrl && (
            <div style={{ textAlign: 'center', marginTop: '2rem' }}>
              <img src={previewUrl} alt="Preview" style={{ maxHeight: '220px', borderRadius: '10px', marginBottom: '1rem', border: '1px solid rgba(255,255,255,0.1)' }} />
              <div>
                <button className="btn" onClick={analyzeImage}>Analyze Scan</button>
              </div>
            </div>
          )}

          {error && (
            <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)', borderRadius: '8px', color: '#fca5a5', fontSize: '0.9rem' }}>
              ⚠ {error}
            </div>
          )}
        </div>
      )}

      {/* ─── Loading Panel ─── */}
      {analyzing && (
        <div className="glass-panel" style={{ textAlign: 'center', maxWidth: '420px', margin: '0 auto' }}>
          <div className="loader" style={{ marginBottom: '1rem' }}>Running EfficientNet-B0 + GradCAM Pipeline…</div>
          <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'var(--accent)', width: '75%', transition: 'width 3s ease' }} />
          </div>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.75rem', fontSize: '0.85rem' }}>Generating explainability heatmap…</p>
        </div>
      )}

      {/* ─── Results Panel ─── */}
      {results && (
        <div>
          {/* Header bar */}
          <div className="glass-panel" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h2 style={{ margin: '0 0 0.5rem 0' }}>Clinical Findings</h2>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <span className={`badge ${results.prediction.referable ? 'danger' : 'success'}`} style={{ background: gradeColor + '22', border: `1px solid ${gradeColor}`, color: gradeColor }}>
                  Grade {results.prediction.grade} — {results.prediction.label}
                </span>
                {tier && <ConfidenceTierBadge tier={tier} />}
                <span style={{ color: 'var(--text-muted)' }}>
                  Confidence: <strong style={{ color: '#e2e8f0' }}>{(results.prediction.confidence * 100).toFixed(1)}%</strong>
                </span>
                <span style={{ color: 'var(--text-muted)' }}>
                  Quality: <strong style={{ color: '#e2e8f0' }}>{results.quality?.status}</strong>
                </span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  ⏱ {results.processing_time_ms?.toFixed(0)} ms
                </span>
                {results.prediction.demo_mode && (
                  <span style={{ fontSize:'0.72rem', color:'#f59e0b', background:'rgba(245,158,11,0.12)', border:'1px solid rgba(245,158,11,0.3)', borderRadius:'4px', padding:'2px 8px' }}>
                    ⚠ DEMO MODE — Model not loaded
                  </span>
                )}
              </div>
            </div>
            <div style={{ display:'flex', gap:'0.75rem', flexWrap:'wrap' }}>
              {results.report_url && (
                <a href={`${API_BASE}${results.report_url}`} target="_blank" rel="noopener noreferrer"
                  style={{ textDecoration:'none' }}>
                  <button className="btn" style={{ background:'rgba(34,197,94,0.2)', border:'1px solid rgba(34,197,94,0.5)' }}>
                    📄 Download Report
                  </button>
                </a>
              )}
              <button className="btn" onClick={reset}>New Screening</button>
            </div>
          </div>

          {/* Referable alert */}
          {results.prediction.referable && (
            <div style={{ marginBottom:'1.5rem', padding:'1rem 1.5rem', background:'rgba(239,68,68,0.12)', border:'1px solid rgba(239,68,68,0.4)', borderRadius:'12px', color:'#fca5a5', fontWeight:600 }}>
              ⚠ REFERABLE DR DETECTED — Ophthalmologist review is required.
            </div>
          )}

          {/* Image grid */}
          <div className="results-grid">
            <ImagePanel title="Original Fundus Image" src={previewUrl} fallback="No preview" />
            <ImagePanel
              title="Grad-CAM Explainability"
              src={results.explainability?.gradcam_url ? `${API_BASE}${results.explainability.gradcam_url}?t=${Date.now()}` : null}
              fallback="Grad-CAM not available (model not loaded)"
            />
            <ImagePanel
              title="Lesion Annotation Map"
              src={results.explainability?.lesion_overlay_url ? `${API_BASE}${results.explainability.lesion_overlay_url}?t=${Date.now()}` : null}
              fallback="Lesion overlay not available"
            />
          </div>

          {/* Class probability breakdown */}
          {results.explainability?.class_probabilities && (
            <div className="glass-panel" style={{ marginTop: '1.5rem' }}>
              <h3 style={{ marginBottom: '1rem' }}>Class Probability Breakdown</h3>
              {Object.entries(results.explainability.class_probabilities).map(([label, prob], i) => (
                <ConfidenceBar key={label} label={label} value={prob} grade={i} />
              ))}
            </div>
          )}

          {/* Lesion analysis */}
          {results.lesions && (
            <div className="glass-panel" style={{ marginTop: '1.5rem' }}>
              <h3 style={{ marginBottom: '0.5rem' }}>Lesion Analysis</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '1rem' }}>
                OpenCV heuristic detection — estimates are image-derived (not hardcoded).
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
                {['microaneurysms', 'hemorrhages', 'exudates'].map((key) => {
                  const val = results.lesions[key] || {};
                  return (
                    <div key={key} style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.04)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {key.replace(/_/g, ' ')}
                      </div>
                      <div style={{ fontSize: '1.4rem', fontWeight: 700, color: val.detected ? '#f59e0b' : '#22c55e', marginTop: '0.25rem' }}>
                        {val.count ?? 0}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {val.detected ? 'Detected' : 'Not detected'} · {((val.confidence ?? 0) * 100).toFixed(0)}% conf
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Clinician Review */}
          <div className="glass-panel" style={{ marginTop: '1.5rem' }}>
            <h3 style={{ marginBottom: '1rem' }}>Clinician Review</h3>
            {reviewSubmitted ? (
              <div style={{ color: '#22c55e', fontWeight: 600 }}>✓ Review submitted successfully.</div>
            ) : (
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <p style={{ color: 'var(--text-muted)', margin: 0 }}>Ophthalmologist decision:</p>
                <button className="btn" disabled={reviewLoading}
                  style={{ background: 'rgba(34,197,94,0.2)', border: '1px solid rgba(34,197,94,0.5)' }}
                  onClick={() => submitReview('CONFIRM')}>
                  ✓ Confirm
                </button>
                <button className="btn" disabled={reviewLoading}
                  style={{ background: 'rgba(239,68,68,0.2)', border: '1px solid rgba(239,68,68,0.5)' }}
                  onClick={() => submitReview('REJECT')}>
                  ✗ Reject
                </button>
                <button className="btn" disabled={reviewLoading}
                  style={{ background: 'rgba(245,158,11,0.2)', border: '1px solid rgba(245,158,11,0.5)' }}
                  onClick={() => submitReview('RECAPTURE')}>
                  ↺ Recapture
                </button>
              </div>
            )}
            <p style={{ marginTop: '0.75rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              ⚠ This AI output is for screening support only. Final diagnosis requires ophthalmologist assessment.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
