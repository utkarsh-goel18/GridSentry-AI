import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowDownRight, ArrowUpRight, Bolt, BrainCircuit, Check, ChevronDown,
  CircleHelp, Clock3, CloudUpload, Cpu, Download, ExternalLink, FileJson2, Gauge,
  LayoutDashboard, Menu, RadioTower, RefreshCw, ShieldCheck, SlidersHorizontal,
  Upload, Waves, X, Zap
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";
import { api, apiBase, type Health, type ModelInfo, type Prediction } from "./api";

const faults = ["AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC"];
const faultNames: Record<string, string> = {
  AG: "Phase A · Ground", BG: "Phase B · Ground", CG: "Phase C · Ground",
  AB: "Phase A · Phase B", BC: "Phase B · Phase C", CA: "Phase C · Phase A",
  ABG: "A + B · Ground", BCG: "B + C · Ground", CAG: "C + A · Ground", ABC: "Three-phase"
};
const waveform = Array.from({ length: 180 }, (_, i) => ({
  t: +(i / 15.36).toFixed(1),
  va: Math.sin(i * .22) * (i < 72 ? 1 : 1.7) + (i < 72 ? 0 : Math.sin(i * .8) * .35),
  vb: Math.sin(i * .22 - 2.09) * (i < 72 ? 1 : 1.35),
  vc: Math.sin(i * .22 + 2.09) * (i < 72 ? 1 : .75)
}));

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const [active, setActive] = useState("overview");
  const [mobileNav, setMobileNav] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictError, setPredictError] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [fileName, setFileName] = useState("");
  const [rawJson, setRawJson] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    setRefreshing(true);
    try {
      const h = await api.health();
      setHealth(h);
      if (h.model_loaded) setModel(await api.modelInfo());
      else setModel(null);
      setConnectionError(h.model_loaded ? "" : "API is reachable, but no model checkpoint is loaded.");
    } catch (e) {
      setHealth(null); setModel(null);
      setConnectionError(e instanceof Error ? e.message : "Could not reach the API.");
    } finally { setRefreshing(false); }
  }
  useEffect(() => { void refresh(); const id = window.setInterval(() => void refresh(), 30000); return () => window.clearInterval(id); }, []);

  const connected = health?.status === "healthy" && health.model_loaded;
  const metricAccuracy = model?.evaluation_summary?.test_accuracy;
  const classData = useMemo(() => faults.map((name) => ({ name, value: prediction?.class_probabilities?.[name] ?? 0 })), [prediction]);

  async function runPrediction() {
    setPredictError(""); setPrediction(null);
    try {
      const parsed = JSON.parse(rawJson);
      const values = Array.isArray(parsed) ? parsed : parsed.waveform;
      if (!Array.isArray(values)) throw new Error("JSON must be a 6×N waveform array or an object with a waveform property.");
      setPredicting(true);
      const result = await api.predict(values, Array.isArray(parsed) ? undefined : parsed.fault_index);
      setPrediction(result);
    } catch (e) { setPredictError(e instanceof Error ? e.message : "Unable to run prediction."); }
    finally { setPredicting(false); }
  }
  function onFile(file?: File) {
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = () => setRawJson(String(reader.result || ""));
    reader.readAsText(file);
  }
  function downloadExample() {
    const sample = { waveform: Array.from({ length: 6 }, (_, c) => Array.from({ length: 512 }, (_, i) => Math.sin(i * .0245 + c * 1.04) * (c < 3 ? 300000 : 12000))), fault_index: 128 };
    const blob = new Blob([JSON.stringify(sample, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const a = document.createElement("a");
    a.href = url; a.download = "waveform-template.json"; a.click(); URL.revokeObjectURL(url);
  }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><RadioTower size={21}/></div><div><strong>GRIDSENTRY<span>AI</span></strong><small>GRID INTELLIGENCE</small></div><button className="mobile-close icon-button" onClick={() => setMobileNav(false)}><X size={18}/></button></div>
      <div className="workspace-label">WORKSPACE</div>
      <nav>
        <button className={active === "overview" ? "nav-item active" : "nav-item"} onClick={() => {setActive("overview");setMobileNav(false)}}><LayoutDashboard size={17}/> Overview <span className="nav-shortcut">01</span></button>
        <button className={active === "inference" ? "nav-item active" : "nav-item"} onClick={() => {setActive("inference");setMobileNav(false)}}><BrainCircuit size={17}/> Fault analysis <span className="nav-shortcut">02</span></button>
        <button className={active === "system" ? "nav-item active" : "nav-item"} onClick={() => {setActive("system");setMobileNav(false)}}><SlidersHorizontal size={17}/> System details <span className="nav-shortcut">03</span></button>
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-status"><span className={connected ? "pulse-dot" : "pulse-dot offline"}/><div><b>{connected ? "System operational" : "System disconnected"}</b><small>{connected ? `API · ${health?.device?.toUpperCase()}` : "Waiting for backend"}</small></div></div>
        <div className="sidebar-foot"><span>GRID PROTECTION / V2</span><span>BUILD 02.01</span></div>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar">
        <div className="topbar-left"><button className="mobile-menu icon-button" onClick={() => setMobileNav(true)}><Menu size={19}/></button><span className="breadcrumb">Workspace</span><span className="crumb-slash">/</span><b>{active === "overview" ? "Overview" : active === "inference" ? "Fault analysis" : "System details"}</b></div>
        <div className="topbar-right"><div className="live-pill"><span className={connected ? "pulse-dot" : "pulse-dot offline"}/>{connected ? "LIVE SYSTEM" : "OFFLINE"}</div><button className="icon-button refresh-btn" onClick={() => void refresh()} title="Refresh system status"><RefreshCw size={16} className={refreshing ? "spin" : ""}/></button><div className="avatar">GS</div></div>
      </header>

      <div className="page-content">
        <section className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line"/> TRANSMISSION INTELLIGENCE PLATFORM</div><h1>{active === "overview" ? <>Grid status<span className="heading-period">.</span></> : active === "inference" ? <>Fault analysis<span className="heading-period">.</span></> : <>System details<span className="heading-period">.</span></>}</h1><p>{active === "overview" ? "A unified view of grid fault intelligence, model health and signal activity." : active === "inference" ? "Submit a six-channel waveform to classify a transmission-line fault." : "Inspect the connected inference service and model configuration."}</p></div><div className="heading-actions"><span className="date-chip"><Clock3 size={14}/>{new Date().toLocaleDateString("en-IN",{day:"2-digit",month:"short",year:"numeric"})}</span><button className="outline-button" onClick={() => void refresh()}><RefreshCw size={14}/> Refresh</button></div></section>

        {connectionError && <div className="notice"><CircleHelp size={17}/><div><b>Backend connection unavailable</b><span>{connectionError} Start the FastAPI service and check VITE_API_URL. Dashboard metrics will remain unreported until the service responds.</span></div><button onClick={() => void refresh()}>Retry</button></div>}

        {active === "overview" && <>
          <section className="metric-grid">
            <Metric icon={<ShieldCheck/>} label="SERVICE STATUS" value={connected ? "Operational" : "Offline"} sub={connected ? "Model loaded and ready" : "Awaiting API connection"} tone={connected ? "green" : "amber"} />
            <Metric icon={<Gauge/>} label="TEST ACCURACY" value={metricAccuracy == null ? "—" : `${(metricAccuracy * 100).toFixed(2)}%`} sub="Reported evaluation artifact" tone="blue" />
            <Metric icon={<Cpu/>} label="MODEL PARAMETERS" value={model ? model.parameters_count.toLocaleString("en-IN") : "—"} sub={model?.model_architecture || "Model metadata unavailable"} tone="violet" />
            <Metric icon={<Activity/>} label="INFERENCE DEVICE" value={health?.device?.toUpperCase() || "—"} sub={health ? "Connected compute target" : "No active connection"} tone="cyan" />
          </section>
          <section className="content-grid">
            <div className="panel waveform-panel"><div className="panel-head"><div><div className="panel-kicker">SIGNAL MONITOR</div><h2>Three-phase voltage profile</h2><p>Illustrative waveform · not a live sensor feed</p></div><span className="tag tag-muted"><Waves size={13}/> 3 PHASE</span></div><div className="chart-legend"><span><i className="legend-a"/> Phase A</span><span><i className="legend-b"/> Phase B</span><span><i className="legend-c"/> Phase C</span></div><div className="wave-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={waveform}><CartesianGrid stroke="#202b3b" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="t" tick={{fill:"#718096",fontSize:10}} tickLine={false} axisLine={false} label={{value:"TIME (MS)",position:"insideBottom",offset:-4,fill:"#718096",fontSize:9}}/><YAxis tick={{fill:"#718096",fontSize:10}} tickLine={false} axisLine={false} width={30}/><Tooltip contentStyle={{background:"#101a2a",border:"1px solid #28364a",borderRadius:8,color:"#e7eef8"}}/><Line type="monotone" dataKey="va" name="Phase A" stroke="#5de0b5" dot={false} strokeWidth={1.8}/><Line type="monotone" dataKey="vb" name="Phase B" stroke="#6ba8ff" dot={false} strokeWidth={1.8}/><Line type="monotone" dataKey="vc" name="Phase C" stroke="#c59aff" dot={false} strokeWidth={1.8}/></LineChart></ResponsiveContainer></div><div className="chart-foot"><span><i className="legend-a"/> Normalised illustrative amplitude</span><span>60 Hz reference</span></div></div>
            <div className="panel distribution-panel"><div className="panel-head"><div><div className="panel-kicker">FAULT TAXONOMY</div><h2>Classification scope</h2><p>Supported fault categories</p></div><span className="tag tag-blue">10 CLASSES</span></div><div className="fault-list">{faults.map((f,i)=><div className="fault-row" key={f}><span className="fault-code">{f}</span><span className="fault-name">{faultNames[f]}</span><span className="fault-index">{String(i+1).padStart(2,"0")}</span></div>)}</div><div className="panel-bottom-note"><Bolt size={14}/> Three-phase voltage + current inputs</div></div>
          </section>
          <section className="bottom-grid"><div className="panel quick-panel"><div className="panel-kicker">QUICK ACTION</div><h2>Run a fault classification</h2><p>Upload a waveform JSON and submit it to the connected inference API.</p><button className="primary-button" onClick={() => setActive("inference")}>Open fault analysis <ArrowUpRight size={15}/></button></div><div className="panel config-panel"><div className="panel-kicker">MODEL INPUT SPECIFICATION</div><div className="spec-grid"><Spec label="CHANNELS" value={model?.input_spec.channels?.join(" · ") || "6 · Va Vb Vc Ia Ib Ic"}/><Spec label="WINDOW LENGTH" value={model ? `${model.input_spec.window_length} samples` : "512 samples"}/><Spec label="SAMPLING RATE" value={model ? `${model.input_spec.sampling_rate_hz.toLocaleString()} Hz` : "15,360 Hz"}/><Spec label="NOMINAL FREQUENCY" value={model ? `${model.input_spec.nominal_freq_hz} Hz` : "60 Hz"}/></div></div></section>
        </>}

        {active === "inference" && <section className="inference-layout"><div className="panel upload-panel"><div className="panel-head"><div><div className="panel-kicker">INPUT / WAVEFORM</div><h2>Submit signal data</h2><p>Provide six channels in [6, N] or [N, 6] format.</p></div><span className="tag tag-muted"><FileJson2 size={13}/> JSON</span></div><div className="dropzone" onClick={() => fileRef.current?.click()} onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault();onFile(e.dataTransfer.files[0]);}}><div className="upload-icon"><CloudUpload size={24}/></div><b>{fileName || "Drop waveform JSON here"}</b><span>or browse files from your device</span><button className="outline-button" type="button" onClick={e => {e.stopPropagation();fileRef.current?.click()}}><Upload size={14}/> Choose file</button><input ref={fileRef} type="file" accept=".json,application/json" hidden onChange={e => onFile(e.target.files?.[0])}/></div><div className="input-divider"><span>OR PASTE JSON</span></div><textarea className="json-input" value={rawJson} onChange={e => setRawJson(e.target.value)} placeholder={'{\n  "waveform": [[...], [...], [...], [...], [...], [...]],\n  "fault_index": 128\n}'} spellCheck={false}/><div className="upload-actions"><button className="text-button" onClick={downloadExample}><Download size={14}/> Download JSON template</button><button className="primary-button" disabled={!connected || !rawJson.trim() || predicting} onClick={() => void runPrediction()}>{predicting ? <><RefreshCw size={14} className="spin"/> Analysing</> : <><Zap size={14}/> Run classification</>}</button></div>{predictError && <div className="error-text">{predictError}</div>}{!connected && <div className="helper-text">Connect a loaded backend model to enable predictions. No sample prediction is fabricated.</div>}</div>
          <div className="panel result-panel"><div className="panel-head"><div><div className="panel-kicker">INFERENCE OUTPUT</div><h2>Classification result</h2><p>Live response from the GridSentry API</p></div><span className="tag tag-muted">RESULT</span></div>{prediction ? <><div className="result-hero"><div className="result-check"><Check size={21}/></div><div><span>PREDICTED FAULT</span><h3>{prediction.predicted_fault}</h3><p>{prediction.fault_description}</p></div><div className="confidence"><b>{(prediction.confidence*100).toFixed(2)}%</b><span>{prediction.confidence_is_calibrated ? "CALIBRATED CONFIDENCE" : "UNCALIBRATED SOFTMAX"}</span>{!prediction.confidence_is_calibrated && <small>Raw model score · not a calibrated probability</small>}{prediction.confidence_is_calibrated && prediction.raw_confidence != null && <small>Raw score: {(prediction.raw_confidence*100).toFixed(2)}%</small>}</div></div><div className="probability-head"><b>Class probabilities</b><span>{prediction.confidence_is_calibrated ? "CALIBRATED MODEL OUTPUT" : "RAW SOFTMAX OUTPUT"}</span></div><div className="probability-list">{classData.map(d=><div className="prob-row" key={d.name}><span>{d.name}</span><div className="prob-track"><i style={{width:`${Math.max(0,Math.min(100,d.value*100))}%`}}/></div><b>{(d.value*100).toFixed(2)}%</b></div>)}</div><div className="result-meta"><span>Inference time <b>{prediction.inference_time_ms.toFixed(2)} ms</b></span><span>Input <b>{prediction.input_type.replace("_"," ")}</b></span></div></> : <div className="empty-result"><div className="empty-icon"><BrainCircuit size={27}/></div><b>Awaiting waveform input</b><p>Submit a valid waveform to see the predicted fault category and probability distribution. Confidence is labelled as calibrated only when a validation-fitted calibration artifact matches the loaded checkpoint.</p></div>}</div></section>}

        {active === "system" && <section className="system-grid"><div className="panel"><div className="panel-kicker">SERVICE CONNECTIVITY</div><h2>Backend health</h2><div className="system-row"><span>API endpoint</span><b>{apiBase}</b></div><div className="system-row"><span>Service status</span><b className={connected ? "good-text" : "warn-text"}>{health?.status || "unreachable"}</b></div><div className="system-row"><span>Model loaded</span><b>{health ? (health.model_loaded ? "Yes" : "No") : "Unknown"}</b></div><div className="system-row"><span>Compute device</span><b>{health?.device || "—"}</b></div><div className="system-row"><span>Last health check</span><b>{health ? new Date(health.timestamp).toLocaleString() : "—"}</b></div><button className="outline-button system-refresh" onClick={() => void refresh()}><RefreshCw size={14}/> Check connection</button></div><div className="panel"><div className="panel-kicker">MODEL CONFIGURATION</div><h2>Inference model</h2><div className="system-row"><span>Model name</span><b>{model?.model_name || "—"}</b></div><div className="system-row"><span>Architecture</span><b>{model?.model_architecture || "—"}</b></div><div className="system-row"><span>Trainable parameters</span><b>{model?.parameters_count.toLocaleString() || "—"}</b></div><div className="system-row"><span>Output classes</span><b>{model?.num_classes || "—"}</b></div><div className="system-row"><span>Input shape</span><b>{model ? `6 × ${model.input_spec.window_length}` : "—"}</b></div><div className="system-row"><span>Sampling rate</span><b>{model?.input_spec.sampling_rate_hz.toLocaleString() || "—"} Hz</b></div></div><div className="panel full-span"><div className="panel-kicker">ENGINEERING NOTE</div><h2>How to interpret reported metrics</h2><p className="body-copy">Accuracy and F1 values are displayed only when supplied by the backend's evaluation artifact. They describe that recorded test evaluation, not guaranteed field performance. The current benchmark consists of simulated 500 kV transmission-line fault recordings. Independent validation on real grid measurements and unseen network topologies is still required before operational deployment.</p></div></section>}

        <footer className="footer"><span>© 2026 GRIDSENTRY AI</span><span>ENGINEERED FOR GRID RELIABILITY <Bolt size={12}/></span><a href="https://github.com/utkarsh-goel18/GridSentry-AI" target="_blank" rel="noreferrer">SOURCE REPOSITORY <ExternalLink size={12}/></a></footer>
      </div>
    </main>
  </div>;
}

function Metric({icon,label,value,sub,tone}:{icon:React.ReactNode;label:string;value:string;sub:string;tone:string}) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><i className={`metric-icon ${tone}`}>{icon}</i></div><strong>{value}</strong><small>{sub}</small></div>;
}
function Spec({label,value}:{label:string;value:string}) { return <div className="spec-item"><span>{label}</span><b>{value}</b></div>; }

export default App;