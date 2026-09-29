import { useState, useEffect } from 'react';
import { translations, languageOptions } from './translations';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
  // ── Theme & Language ────────────────────────────────────────────────────────
  const [theme, setTheme] = useState(() => localStorage.getItem('dr-theme') || 'dark');
  const [lang, setLang] = useState(() => localStorage.getItem('dr-lang') || 'en');
  const t = translations[lang] || translations['en'];

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('dr-theme', theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('dr-lang', lang);
  }, [lang]);

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  // toggleLang is removed because we now use a dropdown.

  // ── App state ────────────────────────────────────────────────────
  const [isRegistered, setIsRegistered] = useState(false);
  const [patientId, setPatientId] = useState(null);
  const [isRegistering, setIsRegistering] = useState(false);
  
  const [fileObj, setFileObj] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [patientName, setPatientName] = useState('');
  const [phoneNumber, setPhoneNumber] = useState('');
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

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!patientName.trim()) {
      setError("Please enter a patient name");
      return;
    }
    
    // Phone number validation: strictly 10 digits
    const phoneRegex = /^\d{10}$/;
    if (!phoneRegex.test(phoneNumber.replace(/\D/g, ''))) {
      setError("Enter a valid 10 digit number");
      return;
    }

    setIsRegistering(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: patientName, phone: phoneNumber })
      });
      if (!resp.ok) throw new Error("Registration failed");
      const data = await resp.json();
      setPatientId(data.patient_id);
      setIsRegistered(true);
    } catch(err) {
      setError(err.message);
    } finally {
      setIsRegistering(false);
    }
  };

  const analyzeImage = async () => {
    if (!fileObj) return;
    setAnalyzing(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', fileObj);
      if (patientId) formData.append('patientId', patientId);
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
    setPatientName('');
    setPhoneNumber('');
    setPatientId(null);
    setIsRegistered(false);
    setResults(null);
    setError(null);
    setReviewSubmitted(false);
  };

  const grade = results?.prediction?.grade;
  const gradeColor = grade !== undefined ? GRADE_COLORS[grade] : '#60a5fa';
  const tier = results?.prediction?.confidence_tier;

  return (
    <div className="dashboard">
      {/* ─── Top Right Controls ─── */}
      <div style={{ position: 'fixed', top: '1rem', right: '1rem', zIndex: 999, display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <select 
          className="theme-toggle" 
          style={{ position: 'static', cursor: 'pointer', paddingRight: '0.5rem', appearance: 'auto' }} 
          value={lang} 
          onChange={(e) => setLang(e.target.value)}
          title="Select language"
        >
          {languageOptions.map(opt => (
            <option key={opt.code} value={opt.code} style={{ background: '#1e293b', color: '#f8fafc' }}>
              {opt.label}
            </option>
          ))}
        </select>
        <button className="theme-toggle" style={{ position: 'static' }} onClick={toggleTheme} title="Toggle theme">
          {theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode'}
        </button>
      </div>

      <header className="header">
        <h1>{t.title}</h1>
        <p style={{ color: 'var(--text-muted)' }}>
          {t.subtitle}
        </p>
      </header>

      {/* ─── Registration Panel ─── */}
      {!isRegistered && !analyzing && (
        <div className="glass-panel" style={{ maxWidth: '400px', margin: '0 auto' }}>
          <h2 style={{ textAlign: 'center', marginBottom: '1.5rem' }}>{t.registerBtn}</h2>
          <form onSubmit={handleRegister} style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>{t.patientName}</label>
              <input 
                type="text" 
                required
                placeholder="e.g. John Doe" 
                value={patientName} 
                onChange={(e) => setPatientName(e.target.value)}
                className="form-input" 
                disabled={isRegistering}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>{t.phoneNumber}</label>
              <input 
                type="tel" 
                placeholder="e.g. 9876543210" 
                value={phoneNumber} 
                onChange={(e) => setPhoneNumber(e.target.value)}
                className="form-input" 
                disabled={isRegistering}
              />
            </div>
            
            {error && (
              <div style={{ padding: '0.75rem', background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)', borderRadius: '8px', color: '#fca5a5', fontSize: '0.85rem' }}>
                ⚠ {error}
              </div>
            )}

            <button type="submit" className="btn" style={{ width: '100%', marginTop: '0.5rem' }} disabled={isRegistering}>
              {isRegistering ? "..." : t.registerBtn}
            </button>
          </form>
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

      {/* ─── Upload Panel ─── */}
      {isRegistered && !results && !analyzing && (
        <div className="glass-panel" style={{ maxWidth: '600px', margin: '0 auto' }}>
          {/* Active Patient Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.2rem' }}>Active Patient</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#22c55e' }}>{patientName}</div>
              {phoneNumber && <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>📞 {phoneNumber}</div>}
            </div>
            <button className="btn" style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.2)', padding: '4px 14px', fontSize: '0.8rem', marginTop: '0' }} onClick={reset}>
              {t.newScreening}
            </button>
          </div>

          {/* Drag-and-drop upload area */}
          <label className="upload-area" onDragOver={(e) => e.preventDefault()} onDrop={handleDrop}>
            <svg width="48" height="48" fill="none" stroke="var(--text-muted)" viewBox="0 0 24 24" style={{ marginBottom: '1rem' }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            <h3>{t.uploadPrompt}</h3>
            <p style={{ color: 'var(--text-muted)', marginTop: '0.4rem' }}>{t.uploadOr}</p>
            <input type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
          </label>

          {previewUrl && (
            <div style={{ textAlign: 'center', marginTop: '2rem' }}>
              <img src={previewUrl} alt="Preview" style={{ maxHeight: '240px', borderRadius: '10px', marginBottom: '1.2rem', border: '1px solid rgba(255,255,255,0.1)' }} />
              <div>
                <button className="btn" onClick={analyzeImage} style={{ padding: '0.75rem 2.5rem', fontSize: '1rem' }}>
                  {t.analyzeScan}
                </button>
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

      {/* ─── Results Panel ─── */}
      {results && (
        <div>
          {/* Header bar */}
          <div className="glass-panel" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h2 style={{ margin: '0 0 0.25rem 0' }}>{t.clinicalFindings}</h2>
              {/* Patient Info Row */}
              {(patientName || phoneNumber) && (
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.75rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  <span>👤 <strong style={{ color: 'var(--body-color)' }}>{patientName}</strong></span>
                  {phoneNumber && <span>📞 <strong style={{ color: 'var(--body-color)' }}>{phoneNumber}</strong></span>}
                </div>
              )}
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <span className={`badge ${results.prediction.referable ? 'danger' : 'success'}`} style={{ background: gradeColor + '22', border: `1px solid ${gradeColor}`, color: gradeColor }}>
                  {t.grade} {results.prediction.grade} — {t.drLabels[results.prediction.label] || results.prediction.label}
                </span>
                {tier && <ConfidenceTierBadge tier={tier} />}
                <span style={{ color: 'var(--text-muted)' }}>
                  {t.confidence}: <strong style={{ color: 'var(--body-color)' }}>{(results.prediction.confidence * 100).toFixed(1)}%</strong>
                </span>
                <span style={{ color: 'var(--text-muted)' }}>
                  {t.quality}: <strong style={{ color: 'var(--body-color)' }}>{results.quality?.status}</strong>
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
                  <button className="btn btn-success">
                    {t.downloadReport}
                  </button>
                </a>
              )}
              <button className="btn" onClick={reset}>{t.newScreening}</button>
            </div>
          </div>

          {/* Referable alert */}
          {results.prediction.referable && (
            <div style={{ marginBottom:'1.5rem', padding:'1rem 1.5rem', background:'rgba(239,68,68,0.12)', border:'1px solid rgba(239,68,68,0.4)', borderRadius:'12px', color:'#fca5a5', fontWeight:600 }}>
              {t.referableAlert}
            </div>
          )}

          {/* Image grid */}
          <div className="results-grid">
            <ImagePanel title={t.originalImage} src={previewUrl} fallback="No preview" />
            <ImagePanel
              title={t.gradCam}
              src={results.explainability?.gradcam_url ? `${API_BASE}${results.explainability.gradcam_url}?t=${Date.now()}` : null}
              fallback="Grad-CAM not available"
            />
            <ImagePanel
              title={t.lesionMap}
              src={results.explainability?.lesion_overlay_url ? `${API_BASE}${results.explainability.lesion_overlay_url}?t=${Date.now()}` : null}
              fallback="Lesion overlay not available"
            />
          </div>

          {/* Class probability breakdown */}
          {results.explainability?.class_probabilities && (
            <div className="glass-panel" style={{ marginTop: '1.5rem' }}>
              <h3 style={{ marginBottom: '1rem' }}>{t.classProbabilities}</h3>
              {Object.entries(results.explainability.class_probabilities).map(([label, prob], i) => (
                <ConfidenceBar key={label} label={t.drLabels[label] || label} value={prob} grade={i} />
              ))}
            </div>
          )}

          {/* Lesion analysis */}
          {results.lesions && (
            <div className="glass-panel" style={{ marginTop: '1.5rem' }}>
              <h3 style={{ marginBottom: '0.5rem' }}>{t.lesionAnalysis}</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '1rem' }}>
                OpenCV heuristic detection — estimates are image-derived (not hardcoded).
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem' }}>
                {['microaneurysms', 'hemorrhages', 'exudates'].map((key) => {
                  const val = results.lesions[key] || {};
                  return (
                    <div key={key} style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.04)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {t[key] || key.replace(/_/g, ' ')}
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
            <h3 style={{ marginBottom: '1rem' }}>{t.clinicianReview}</h3>
            {reviewSubmitted ? (
              <div style={{ color: '#22c55e', fontWeight: 600 }}>✓ Review submitted.</div>
            ) : (
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <button className="btn btn-success" disabled={reviewLoading}
                  onClick={() => submitReview('CONFIRM')}>
                  {t.confirm}
                </button>
                <button className="btn btn-danger" disabled={reviewLoading}
                  onClick={() => submitReview('REJECT')}>
                  {t.reject}
                </button>
                <button className="btn btn-warning" disabled={reviewLoading}
                  onClick={() => submitReview('RECAPTURE')}>
                  {t.recapture}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
