import { useState, useRef, useCallback, useEffect } from "react";
import CoralLogo from "./components/CoralLogo"; 
import Header from "./components/Header";
import Footer from "./components/Footer";

const API_URL = "http://localhost:8000/predict";
const STATS_URL = "http://localhost:8000/analytics/stats";
const HISTORY_URL = "http://localhost:8000/history";

const CLASS_CONFIG = {
  healthy_corals: {
    color: "#00e5a0", bg: "rgba(0,229,160,0.08)",
    border: "rgba(0,229,160,0.3)", glow: "0 0 24px rgba(0,229,160,0.25)",
    label: "HEALTHY", icon: "◉",
  },
  bleached_corals: {
    color: "#f5c400", bg: "rgba(245,196,0,0.08)",
    border: "rgba(245,196,0,0.3)", glow: "0 0 24px rgba(245,196,0,0.25)",
    label: "BLEACHED", icon: "◎",
  },
  dead: {
    color: "#ff4d6d", bg: "rgba(255,77,109,0.08)",
    border: "rgba(255,77,109,0.3)", glow: "0 0 24px rgba(255,77,109,0.25)",
    label: "DEAD", icon: "○",
  },
};

const DEFAULT_CFG = {
  color: "#00b4dc", bg: "rgba(0,180,220,0.08)",
  border: "rgba(0,180,220,0.3)", glow: "0 0 24px rgba(0,180,220,0.2)",
  label: "UNKNOWN", icon: "◌",
};

const URGENCY_CONFIG = {
  low:      { color: "#00e5a0", text: "LOW URGENCY"      },
  high:     { color: "#f5c400", text: "HIGH URGENCY"     },
  critical: { color: "#ff4d6d", text: "CRITICAL URGENCY" },
};

const S = {
  root: {
    minHeight: "100vh",
    background: "#050d1a",
    backgroundImage: "radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,180,220,0.12) 0%, transparent 70%)",
    fontFamily: "'DM Mono', 'Fira Code', 'Courier New', monospace",
    color: "#c8d8e8", overflowX: "hidden",
  },
  scanlines: {
    position: "fixed", inset: 0,
    backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px)",
    pointerEvents: "none", zIndex: 0,
  },
  inner: { position: "relative", zIndex: 1, maxWidth: 780, margin: "0 auto", padding: "48px 24px 80px" },
  header: { marginBottom: 32 },
  eyebrow: {
    fontSize: 10, letterSpacing: "0.25em", color: "#00b4dc",
    marginBottom: 10, display: "flex", alignItems: "center", gap: 8,
  },
  dot: {
    width: 5, height: 5, borderRadius: "50%",
    background: "#00b4dc", boxShadow: "0 0 8px #00b4dc",
    animation: "pulse 2s ease-in-out infinite",
  },
  title: {
    fontSize: "clamp(26px,5vw,40px)", fontWeight: 700,
    color: "#e8f4ff", letterSpacing: "-0.02em", lineHeight: 1.1, margin: "0 0 10px",
  },
  sub: { fontSize: 13, color: "#5a7a9a", letterSpacing: "0.04em", lineHeight: 1.6 },
  
  // --- Analytics Dashboard Styles ---
  statsContainer: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
    gap: "16px",
    marginBottom: "32px",
    animation: "scanIn 0.6s ease forwards",
  },
  statCard: {
    background: "rgba(0,180,220,0.03)",
    border: "1px solid rgba(0,180,220,0.12)",
    borderRadius: "12px",
    padding: "16px",
    textAlign: "center",
  },
  statVal: { fontSize: "22px", fontWeight: "700", color: "#00b4dc", display: "block", marginBottom: "4px" },
  statLabel: { fontSize: "9px", letterSpacing: "0.15em", color: "#5a7a9a", textTransform: "uppercase" },

  dropzone: (drag, hasPreview) => ({
    border: `1px solid ${drag ? "#00b4dc" : "rgba(0,180,220,0.15)"}`,
    borderRadius: 12,
    background: drag ? "rgba(0,180,220,0.06)" : "rgba(255,255,255,0.02)",
    padding: hasPreview ? 0 : "52px 24px",
    cursor: "pointer", transition: "all 0.2s ease",
    position: "relative", overflow: "hidden",
    boxShadow: drag ? "0 0 30px rgba(0,180,220,0.15)" : "none",
  }),
  corner: pos => ({
    position: "absolute", width: 14, height: 14,
    borderColor: "#00b4dc", borderStyle: "solid", borderWidth: 0, opacity: 0.5,
    ...(pos === "tl" && { top: 8, left: 8, borderTopWidth: 1.5, borderLeftWidth: 1.5 }),
    ...(pos === "tr" && { top: 8, right: 8, borderTopWidth: 1.5, borderRightWidth: 1.5 }),
    ...(pos === "bl" && { bottom: 8, left: 8, borderBottomWidth: 1.5, borderLeftWidth: 1.5 }),
    ...(pos === "br" && { bottom: 8, right: 8, borderBottomWidth: 1.5, borderRightWidth: 1.5 }),
  }),
  dropContent: { display: "flex", flexDirection: "column", alignItems: "center", gap: 12, pointerEvents: "none" },
  uploadIcon: {
    width: 44, height: 44, borderRadius: "50%",
    border: "1px solid rgba(0,180,220,0.25)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 18, color: "#00b4dc",
  },
  dropTitle: { fontSize: 14, color: "#8aaabf", letterSpacing: "0.05em" },
  dropSub:   { fontSize: 11, color: "#3a5a7a", letterSpacing: "0.08em" },
  previewWrap: { position: "relative" },
  previewImg: { width: "100%", maxHeight: 340, objectFit: "cover", borderRadius: 11, display: "block", filter: "brightness(0.9)" },
  previewOverlay: {
    position: "absolute", inset: 0, borderRadius: 11,
    background: "linear-gradient(to bottom, transparent 50%, rgba(5,13,26,0.85) 100%)",
    display: "flex", alignItems: "flex-end", padding: "16px 18px",
  },
  previewMeta: { fontSize: 11, color: "rgba(200,216,232,0.7)", letterSpacing: "0.06em" },
  changeBtn: {
    position: "absolute", top: 12, right: 12, fontSize: 11, letterSpacing: "0.1em",
    color: "#00b4dc", background: "rgba(5,13,26,0.8)",
    border: "1px solid rgba(0,180,220,0.25)", borderRadius: 6,
    padding: "5px 10px", cursor: "pointer",
  },
  analyseBtn: disabled => ({
    width: "100%", marginTop: 16, padding: "15px 24px",
    background: disabled ? "rgba(0,180,220,0.05)" : "rgba(0,180,220,0.1)",
    border: `1px solid ${disabled ? "rgba(0,180,220,0.1)" : "rgba(0,180,220,0.35)"}`,
    borderRadius: 10, color: disabled ? "#3a5a7a" : "#00b4dc",
    fontSize: 12, letterSpacing: "0.18em", fontFamily: "inherit",
    cursor: disabled ? "not-allowed" : "pointer", transition: "all 0.2s",
    display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
  }),
  loadingWrap: {
    marginTop: 24, padding: "28px 24px",
    border: "1px solid rgba(0,180,220,0.12)", borderRadius: 12,
    background: "rgba(0,180,220,0.03)",
    display: "flex", flexDirection: "column", alignItems: "center", gap: 14,
  },
  loadingText: { fontSize: 11, letterSpacing: "0.2em", color: "#3a6a8a", animation: "blink 1.2s step-end infinite" },
  resultCard: cfg => ({
    marginTop: 24, border: `1px solid ${cfg.border}`, borderRadius: 12,
    background: cfg.bg, boxShadow: cfg.glow, overflow: "hidden",
    animation: "scanIn 0.4s ease forwards",
  }),
  resultHeader: cfg => ({
    padding: "20px 24px 16px", borderBottom: `1px solid ${cfg.border}`,
    display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16,
  }),
  resultLabel: cfg => ({
    fontSize: 22, fontWeight: 700, color: cfg.color, letterSpacing: "0.08em",
    display: "flex", alignItems: "center", gap: 10,
  }),
  confidencePill: cfg => ({
    fontSize: 11, letterSpacing: "0.12em", color: cfg.color,
    background: `${cfg.color}20`, border: `1px solid ${cfg.border}`,
    borderRadius: 20, padding: "4px 12px", whiteSpace: "nowrap",
  }),
  probSection: { padding: "18px 24px", borderBottom: "1px solid rgba(255,255,255,0.04)" },
  probLabel: { fontSize: 10, letterSpacing: "0.2em", color: "#3a5a7a", marginBottom: 14 },
  probRow: { display: "flex", alignItems: "center", gap: 10, marginBottom: 10 },
  probName: { fontSize: 11, letterSpacing: "0.1em", color: "#5a7a9a", width: 80, flexShrink: 0 },
  probTrack: { flex: 1, height: 4, background: "rgba(255,255,255,0.05)", borderRadius: 2, overflow: "hidden" },
  probFill: (pct, color) => ({
    height: "100%", width: `${pct}%`, background: color, borderRadius: 2,
    transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)", boxShadow: `0 0 8px ${color}55`,
  }),
  probPct: { fontSize: 11, color: "#5a7a9a", width: 42, textAlign: "right", flexShrink: 0 },
  actionsSection: { padding: "18px 24px" },
  urgencyBadge: u => ({
    display: "inline-flex", alignItems: "center", gap: 6,
    fontSize: 10, letterSpacing: "0.18em", color: u.color,
    border: `1px solid ${u.color}44`, borderRadius: 4, padding: "3px 10px", marginBottom: 14,
  }),
  urgencyDot: u => ({ width: 4, height: 4, borderRadius: "50%", background: u.color, boxShadow: `0 0 6px ${u.color}` }),
  statusText: { fontSize: 13, color: "#8aaabf", marginBottom: 14, lineHeight: 1.5 },
  actionList: { listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: 8 },
  actionItem: { fontSize: 12, color: "#5a7a9a", letterSpacing: "0.02em", lineHeight: 1.55, display: "flex", gap: 10 },
  actionBullet: { color: "#00b4dc", flexShrink: 0, marginTop: 1, fontSize: 10 },
  metaFooter: { padding: "12px 24px", borderTop: "1px solid rgba(255,255,255,0.04)", display: "flex", gap: 20, flexWrap: "wrap" },
  metaItem: { fontSize: 10, letterSpacing: "0.12em", color: "#2a4a6a", display: "flex", gap: 6 },
  metaVal: { color: "#4a7a9a" },
  errorBox: {
    marginTop: 20, padding: "16px 20px",
    border: "1px solid rgba(255,77,109,0.25)", borderRadius: 10,
    background: "rgba(255,77,109,0.05)",
    fontSize: 12, color: "#ff8099", letterSpacing: "0.04em", lineHeight: 1.6,
    display: "flex", gap: 10,
  },
};

const injectStyles = () => {
  if (document.getElementById("coral-kf")) return;
  const s = document.createElement("style");
  s.id = "coral-kf";
  s.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;700&display=swap');
    @keyframes pulse   { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.85)} }
    @keyframes blink   { 0%,100%{opacity:1} 50%{opacity:.2} }
    @keyframes spin    { to{transform:rotate(360deg)} }
    @keyframes scanIn  { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  `;
  document.head.appendChild(s);
};

function Spinner() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" style={{ animation: "spin 1s linear infinite" }}>
      <circle cx="16" cy="16" r="12" fill="none" stroke="rgba(0,180,220,0.15)" strokeWidth="2"/>
      <path d="M16 4 A12 12 0 0 1 28 16" fill="none" stroke="#00b4dc" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

export default function CoralHealthMonitor() {
  injectStyles();
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [stats, setStats]     = useState(null); // --- NEW STATS STATE ---
  const [history, setHistory] = useState([]);
  const [error, setError]     = useState(null);
  const inputRef              = useRef(null);

  // --- FETCH ANALYTICS & HISTORY ---
  const refreshData = useCallback(async () => {
    try {
      const [statsRes, historyRes] = await Promise.all([
        fetch(STATS_URL),
        fetch(HISTORY_URL)
      ]);
      setStats(await statsRes.json());
      setHistory(await historyRes.json());
    } catch (err) {
      console.error("Sync error:", err);
    }
  }, []);

  useEffect(() => {
    refreshData();
  }, [result, refreshData]);


  const acceptFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) { setError("Please upload an image."); return; }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(null);
  }, []);

  const onInputChange  = e  => acceptFile(e.target.files[0]);
  const onDrop         = e  => { e.preventDefault(); setDrag(false); acceptFile(e.dataTransfer.files[0]); };
   const onDragOver     = e  => { e.preventDefault(); setDrag(true); };
   const onDragLeave    = () => setDrag(false);
   const openFilePicker = () => inputRef.current?.click();

  // --- VALIDATION LOOP HANDLER ---
  const handleVerify = async (id, label) => {
    try {
      const res = await fetch(`http://localhost:8000/verify/${id}?correct_label=${label}`, { method: 'PATCH' });
      if (res.ok) refreshData();
    } catch (err) {
      console.error("Verification failed", err);
    }
  };

  const analyse = async () => {
    if (!file || loading) return;
    setLoading(true); setResult(null); setError(null);

    // Get Geospatial Data
    let lat = null, lon = null;
    if ("geolocation" in navigator) {
      const pos = await new Promise((res) => navigator.geolocation.getCurrentPosition(res, () => res(null)));
      if (pos) { lat = pos.coords.latitude; lon = pos.coords.longitude; }
    }
    
    const body = new FormData();
    body.append("file", file);
    if(lat) body.append("lat", lat);
    if(lon) body.append("lon", lon);
    try {
      const res = await fetch(API_URL, { method: "POST", body });
      if (!res.ok) 
        throw new Error("Inference failed");
      setResult(await res.json());
    
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const cfg        = result ? (CLASS_CONFIG[result.label] || DEFAULT_CFG) : null;
  const urgencyCfg = result ? (URGENCY_CONFIG[result.urgency] || URGENCY_CONFIG.low) : null;

  return (
    <div style={S.root}>
      <div style={S.scanlines} />
      <div style={S.inner}>
        <Header S={S} />
        {/* --- ANALYTICS DASHBOARD --- */}
        {stats && (
           <div style={S.statsContainer}>
            <div style={S.statCard}>
              <span style={S.statLabel}>System Scans</span>
              <span style={S.statVal}>{stats.total_scans}</span>
            </div>
            <div style={S.statCard}>
              <span style={S.statLabel}>Bleaching Rate</span>
              <span style={S.statVal}>{stats.bleaching_rate}</span>
            </div>
            <div style={S.statCard}>
              <span style={S.statLabel}>Accuracy verified</span>
              <span style={S.statVal}>{history.filter(h => h.is_verified).length}</span>
            </div>
          </div>
        )}

        {/* Drop zone */}
        <div style={S.dropzone(isDragging, !!preview)}
          onClick={!preview ? openFilePicker : undefined}
          onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}>
          {["tl","tr","bl","br"].map(p => <div key={p} style={S.corner(p)} />)}
          {preview ? (
            <div style={S.previewWrap}>
              <img src={preview} alt="Coral preview" style={S.previewImg} />
              <div style={S.previewOverlay}>
                <span style={S.previewMeta}>{file.name} · {(file.size/1024).toFixed(0)} KB</span>
              </div>
              <button style={S.changeBtn} onClick={e => { e.stopPropagation(); openFilePicker(); }}>CHANGE ↑</button>
            </div>
          ) : (
            <div style={S.dropContent}>
              <div style={S.uploadIcon}>↑</div>
              <span style={S.dropTitle}>DROP IMAGE HERE OR CLICK TO BROWSE</span>
              <span style={S.dropSub}>JPEG · PNG · WEBP · MAX 10 MB</span>
            </div>
          )}
        </div>
        <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp"
          style={{ display: "none" }} onChange={onInputChange} />

        {/* Analyse button */}
        <button style={S.analyseBtn(!file || loading)} onClick={analyse} disabled={!file || loading}>
          {loading ? <><Spinner /> RUNNING HYBRID AI</> : "◈  START ANALYSIS"}
        </button>

        {loading && (
          <div style={S.loadingWrap}>
            <Spinner />
            <span style={S.loadingText}>CONSULTING GEMINI AI FOR RECOMMENDATIONS</span>
          </div>
        )}

        {error && <div style={S.errorBox}><span>⚠</span>{error}</div>}

        {/* Result card */}
        {result && cfg && (
          <div style={S.resultCard(cfg)}>
            <div style={S.resultHeader(cfg)}>
              <div style={S.resultLabel(cfg)}>
                <span style={{ textShadow: `0 0 10px ${cfg.color}` }}>{cfg.icon}</span>
                {cfg.label}
              </div>
              <span style={S.confidencePill(cfg)}>{result.confidence_pct} CONFIDENCE</span>
            </div>

            <div style={S.probSection}>
              <div style={S.probLabel}>VISION MODEL PROBABILITIES</div>
              {result.probabilities && Object.entries(result.probabilities).map(([cls, prob]) => {
                const c = CLASS_CONFIG[cls] || DEFAULT_CFG;
                return (
                  <div key={cls} style={S.probRow}>
                    <span style={S.probName}>{cls.replace('_', ' ').toUpperCase()}</span>
                    <div style={S.probTrack}><div style={S.probFill(prob * 100, c.color)} /></div>
                    <span style={S.probPct}>{(prob * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>

            <div style={S.actionsSection}>
              <div style={S.urgencyBadge(urgencyCfg)}>
                <div style={S.urgencyDot(urgencyCfg)} />
                {urgencyCfg.text}
              </div>
              <p style={S.statusText}>{result.status}</p>
              <ul style={S.actionList}>
                {result.actions.map((a, i) => (
                  <li key={i} style={S.actionItem}><span style={S.actionBullet}>›</span>{a}</li>
                ))}
              </ul>
            </div>

            <div style={S.metaFooter}>
              <span style={S.metaItem}>INFERENCE <span style={S.metaVal}>{result.inference_ms} MS</span></span>
              <span style={S.metaItem}>LLM <span style={S.metaVal}>GEMINI-2.5-FLASH-LITE</span></span>
              <span style={S.metaItem}>LOC <span style={S.metaVal}>SEC-A REEF</span></span>
            </div>
          </div>
        )}

        {/* --- GEOSPATIAL HISTORY & VALIDATION LOOP UI --- */}
        <div style={{ marginTop: '40px' }}>
          <div style={S.probLabel}>GEOSPATIAL LOGS & VALIDATION</div>
          <div style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(0,180,220,0.1)', overflow: 'hidden' }}>
            {history.map((item) => (
              <div key={item.id} style={{ padding: '16px 20px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '16px' }}>
                <div>
                  <div style={{ fontSize: '12px', color: CLASS_CONFIG[item.prediction_label]?.color || '#fff', fontWeight: 'bold' }}>
                    {item.prediction_label.replace('_', ' ').toUpperCase()}
                  </div>
                  <div style={{ fontSize: '10px', color: '#4a6a8a', marginTop: '2px' }}>
                    {item.filename}
                  </div>

                  <div style={{ fontSize: '10px', color: '#3a5a7a', marginTop: '4px' }}>
                    {item.latitude
                    ? `COORDS: ${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)}`
                    : 'GPS DATA UNAVAILABLE'}
                  </div>
                </div>
                
                {item.is_verified ? (
  <div style={{ textAlign: 'right' }}>
    <span style={{ fontSize: '9px', color: '#00e5a0', letterSpacing: '0.1em' }}>✓ VERIFIED</span>
    <div style={{ fontSize: '9px', color: '#3a5a7a', marginTop: '4px' }}>
      {item.verified_label.replace('_', ' ').toUpperCase()}
    </div>
  </div>
) : (
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px' }}>
    <span style={{ fontSize: '9px', color: '#3a5a7a', letterSpacing: '0.1em' }}>WAS THIS CORRECT?</span>
    <div style={{ display: 'flex', gap: '6px' }}>
      {/* ✓ Confirm model was right — passes model's own label */}
      <button
        onClick={() => handleVerify(item.id, item.prediction_label)}
        style={{
          fontSize: '9px', padding: '5px 10px', cursor: 'pointer',
          background: 'transparent', borderRadius: '4px',
          border: '1px solid #00e5a0', color: '#00e5a0',
        }}
      >
        ✓ CORRECT
      </button>

      {/* ✗ Human disagrees — passes the OPPOSITE label */}
      <button
        onClick={() => handleVerify(
          item.id,
          item.prediction_label === 'healthy_corals' ? 'bleached_corals' : 'healthy_corals'
        )}
        style={{
          fontSize: '9px', padding: '5px 10px', cursor: 'pointer',
          background: 'transparent', borderRadius: '4px',
          border: '1px solid #f5c400', color: '#f5c400',
        }}
      >
        ✗ WRONG →{' '}
        {(item.prediction_label === 'healthy_corals'
          ? 'BLEACHED'
          : 'HEALTHY')}
      </button>
    </div>
  </div>
)}
              </div>
            ))}
          </div>
        </div>

        <Footer S={S} />
      </div>
    </div>
  );
}