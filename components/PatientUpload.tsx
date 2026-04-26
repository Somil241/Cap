/**
 * PatientUpload.tsx  →  drop into: components/PatientUpload.tsx
 */

import React, { useCallback, useRef, useState } from 'react';

interface ShapFeature {
  name: string;
  shap: number;
  value?: number;
}

interface PredictionResult {
  patient_id: string;
  sepsis_risk: number;
  risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  alert_6h: boolean;
  top_features: ShapFeature[];
  sofa_total?: number;
  vent_prob?: number;
}

const VITAL_FIELDS = [
  { key: 'heart_rate',  label: 'Heart Rate',   unit: 'bpm',    placeholder: '80'   },
  { key: 'sbp',         label: 'Systolic BP',  unit: 'mmHg',   placeholder: '120'  },
  { key: 'dbp',         label: 'Diastolic BP', unit: 'mmHg',   placeholder: '80'   },
  { key: 'map',         label: 'MAP',          unit: 'mmHg',   placeholder: '93'   },
  { key: 'spo2',        label: 'SpO₂',         unit: '%',      placeholder: '97'   },
  { key: 'resp_rate',   label: 'Resp. Rate',   unit: '/min',   placeholder: '16'   },
  { key: 'temperature', label: 'Temperature',  unit: '°C',     placeholder: '37.0' },
  { key: 'gcs_total',   label: 'GCS Total',    unit: '/15',    placeholder: '15'   },
] as const;

const LAB_FIELDS = [
  { key: 'lactate',    label: 'Lactate',    unit: 'mmol/L',  placeholder: '1.0' },
  { key: 'creatinine', label: 'Creatinine', unit: 'mg/dL',   placeholder: '0.9' },
  { key: 'bilirubin',  label: 'Bilirubin',  unit: 'mg/dL',   placeholder: '0.6' },
  { key: 'wbc',        label: 'WBC',        unit: '×10³/µL', placeholder: '9.0' },
  { key: 'platelet',   label: 'Platelets',  unit: '×10³/µL', placeholder: '220' },
  { key: 'pao2_fio2',  label: 'PaO₂/FiO₂', unit: 'ratio',   placeholder: '400' },
] as const;

const RISK_TEXT:   Record<string, string> = { LOW: 'text-emerald-400', MODERATE: 'text-amber-400',  HIGH: 'text-orange-400', CRITICAL: 'text-red-400'  };
const RISK_BORDER: Record<string, string> = { LOW: 'border-emerald-500/30 bg-emerald-500/5', MODERATE: 'border-amber-500/30 bg-amber-500/5', HIGH: 'border-orange-500/30 bg-orange-500/5', CRITICAL: 'border-red-500/30 bg-red-500/5' };
const RISK_BAR:    Record<string, string> = { LOW: 'bg-emerald-500', MODERATE: 'bg-amber-500', HIGH: 'bg-orange-500', CRITICAL: 'bg-red-500' };

const API = 'http://localhost:8765';

// ── Prediction card ───────────────────────────────────────────────────────────

const PredictionCard: React.FC<{ pred: PredictionResult }> = ({ pred }) => {
  const rl = pred.risk_level;
  return (
    <div className={`rounded-2xl border p-5 space-y-4 ${RISK_BORDER[rl]}`}>
      <div className="flex justify-between items-start gap-2">
        <div className="min-w-0">
          <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Patient</p>
          <p className="text-sm font-black text-white mt-0.5 truncate">{pred.patient_id}</p>
        </div>
        <div className="text-right shrink-0">
          <p className={`text-[10px] font-black uppercase ${RISK_TEXT[rl]}`}>
            {rl}{pred.alert_6h ? ' · ⚠ 6h Alert' : ''}
          </p>
          <p className="text-3xl font-black text-white leading-none">
            {(pred.sepsis_risk * 100).toFixed(1)}<span className="text-sm text-slate-400">%</span>
          </p>
        </div>
      </div>

      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${RISK_BAR[rl]}`} style={{ width: `${pred.sepsis_risk * 100}%` }} />
      </div>

      {(pred.sofa_total !== undefined || pred.vent_prob !== undefined) && (
        <div className="grid grid-cols-2 gap-2">
          {pred.sofa_total !== undefined && (
            <div className="bg-slate-900/60 rounded-xl p-3">
              <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest">SOFA</p>
              <p className="text-xl font-black text-white">{pred.sofa_total}<span className="text-xs text-slate-500">/24</span></p>
            </div>
          )}
          {pred.vent_prob !== undefined && (
            <div className="bg-slate-900/60 rounded-xl p-3">
              <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Vent Risk</p>
              <p className="text-xl font-black text-white">{(pred.vent_prob * 100).toFixed(0)}<span className="text-xs text-slate-500">%</span></p>
            </div>
          )}
        </div>
      )}

      {pred.top_features.length > 0 && (
        <div>
          <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-2">SHAP Drivers</p>
          <div className="space-y-1.5">
            {pred.top_features.slice(0, 5).map((f, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-[9px] text-slate-400 w-28 shrink-0 truncate">{f.name.replace(/_/g, ' ')}</span>
                <div className="flex-1 bg-slate-800 rounded-full h-1 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${f.shap >= 0 ? 'bg-red-400' : 'bg-emerald-400'}`}
                    style={{ width: `${Math.min(100, Math.abs(f.shap) * 300)}%` }}
                  />
                </div>
                <span className={`text-[9px] font-black w-14 text-right ${f.shap >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                  {f.shap >= 0 ? '+' : ''}{f.shap.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
          <p className="text-[8px] text-slate-600 mt-2">Red ↑ risk · Green ↓ risk</p>
        </div>
      )}
    </div>
  );
};

// ── Main component ────────────────────────────────────────────────────────────

type Mode = 'csv' | 'manual';

const PatientUpload: React.FC = () => {
  const [mode, setMode]               = useState<Mode>('csv');
  const [dragging, setDragging]       = useState(false);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [rowCount, setRowCount]       = useState<number | null>(null);
  const [patientId, setPatientId]     = useState('PT-NEW-001');
  const [form, setForm]               = useState<Record<string, string>>({});
  const fileRef                        = useRef<HTMLInputElement>(null);

  const reset = () => { setPredictions([]); setError(null); setRowCount(null); };

  const submitCsv = useCallback(async (file: File) => {
    if (!file.name.endsWith('.csv')) { setError('Only .csv files are accepted.'); return; }
    setLoading(true); reset();
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch(`${API}/api/predict/csv`, { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setRowCount(data.row_count ?? 0);
      setPredictions(data.predictions ?? []);
    } catch (e: any) {
      setError(e.message ?? 'Upload failed. Is the backend running?');
    } finally { setLoading(false); }
  }, []);

  const submitManual = async () => {
    setLoading(true); reset();
    const body: Record<string, any> = { patient_id: patientId };
    Object.entries(form).forEach(([k, v]) => { if (v.trim()) body[k] = parseFloat(v); });
    try {
      const res  = await fetch(`${API}/api/predict/manual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setRowCount(1);
      if (data.prediction) setPredictions([data.prediction]);
    } catch (e: any) {
      setError(e.message ?? 'Prediction failed. Is the backend running?');
    } finally { setLoading(false); }
  };

  const downloadTemplate = async () => {
    try {
      const data = await fetch(`${API}/api/predict/template`).then(r => r.json());
      const cols: string[] = data.columns;
      const rows: Record<string, any>[] = data.example_rows;
      const csv = [cols.join(','), ...rows.map(r => cols.map((c: string) => r[c] ?? '').join(','))].join('\n');
      const a = Object.assign(document.createElement('a'), {
        href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })),
        download: 'icu_patient_template.csv',
      });
      a.click();
    } catch { setError('Could not fetch template — is the backend running?'); }
  };

  const setField = (key: string, v: string) => setForm(f => ({ ...f, [key]: v }));

  return (
    <div className="bg-slate-900/30 rounded-[2rem] border border-white/5 shadow-2xl overflow-hidden">

      {/* Header */}
      <div className="p-6 border-b border-white/5 bg-slate-900/60 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-black text-white tracking-tight">Patient Data Ingestion</h2>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mt-0.5">
            Upload CSV or enter vitals → XGBoost + SHAP
          </p>
        </div>
        <div className="flex gap-1 bg-slate-950/60 border border-white/10 rounded-xl p-1 self-start sm:self-auto">
          {(['csv', 'manual'] as Mode[]).map(m => (
            <button key={m} onClick={() => { setMode(m); reset(); }}
              className={`px-4 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${
                mode === m ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20' : 'text-slate-500 hover:text-slate-300'
              }`}>
              {m === 'csv' ? '📂  CSV' : '✏️  Manual'}
            </button>
          ))}
        </div>
      </div>

      <div className="p-6 grid grid-cols-1 xl:grid-cols-2 gap-6">

        {/* LEFT: input */}
        <div className="space-y-4">

          {mode === 'csv' && (
            <>
              <div
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files?.[0]; if (f) submitCsv(f); }}
                onClick={() => fileRef.current?.click()}
                className={`cursor-pointer border-2 border-dashed rounded-2xl p-10 text-center transition-all select-none ${
                  dragging ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
                           : 'border-white/10 hover:border-indigo-500/50 hover:bg-indigo-500/5'
                }`}>
                <div className="text-5xl mb-3 pointer-events-none">{dragging ? '🎯' : '📋'}</div>
                <p className="text-sm font-black text-slate-300 pointer-events-none">
                  {dragging ? 'Release to upload' : 'Drag & drop a CSV'}
                </p>
                <p className="text-[10px] text-slate-600 mt-1 pointer-events-none">or click to browse</p>
                <input ref={fileRef} type="file" accept=".csv" className="hidden"
                  onChange={e => { const f = e.target.files?.[0]; if (f) submitCsv(f); e.target.value = ''; }} />
              </div>

              <button onClick={downloadTemplate}
                className="w-full text-[10px] font-black uppercase tracking-widest text-indigo-400 hover:text-indigo-300 border border-indigo-500/30 hover:border-indigo-400/60 rounded-xl py-2.5 transition-all">
                ↓ Download CSV Template
              </button>

              <div className="bg-slate-950/40 rounded-xl p-4">
                <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-2">Expected columns (any subset)</p>
                <div className="flex flex-wrap gap-1.5">
                  {['patient_id', 'charttime', ...VITAL_FIELDS.map(f => f.key), ...LAB_FIELDS.map(f => f.key)].map(c => (
                    <span key={c} className="text-[9px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded-md">{c}</span>
                  ))}
                </div>
                <p className="text-[9px] text-slate-600 mt-2">
                  Missing columns are zero-filled. Include <code className="bg-slate-800 px-1 rounded">charttime</code> for rolling features.
                </p>
              </div>
            </>
          )}

          {mode === 'manual' && (
            <div className="space-y-5">
              <div>
                <label className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Patient ID</label>
                <input value={patientId} onChange={e => setPatientId(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500 transition"
                  placeholder="PT-NEW-001" />
              </div>

              <div>
                <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-2">Vital Signs</p>
                <div className="grid grid-cols-2 gap-2">
                  {VITAL_FIELDS.map(f => (
                    <div key={f.key}>
                      <label className="text-[9px] text-slate-500 font-bold">
                        {f.label} <span className="text-slate-600">({f.unit})</span>
                      </label>
                      <input type="number" step="any" value={form[f.key] ?? ''} placeholder={f.placeholder}
                        onChange={e => setField(f.key, e.target.value)}
                        className="mt-0.5 w-full bg-slate-900 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 transition placeholder:text-slate-700" />
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-2">Laboratory Values</p>
                <div className="grid grid-cols-2 gap-2">
                  {LAB_FIELDS.map(f => (
                    <div key={f.key}>
                      <label className="text-[9px] text-slate-500 font-bold">
                        {f.label} <span className="text-slate-600">({f.unit})</span>
                      </label>
                      <input type="number" step="any" value={form[f.key] ?? ''} placeholder={f.placeholder}
                        onChange={e => setField(f.key, e.target.value)}
                        className="mt-0.5 w-full bg-slate-900 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 transition placeholder:text-slate-700" />
                    </div>
                  ))}
                </div>
              </div>

              <button onClick={submitManual} disabled={loading}
                className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white text-[10px] font-black uppercase tracking-widest py-3 rounded-xl transition-all shadow-lg shadow-indigo-600/20">
                {loading ? 'Running…' : '▶  Run Sepsis Prediction'}
              </button>
            </div>
          )}
        </div>

        {/* RIGHT: results */}
        <div className="space-y-4 min-h-[200px]">

          {loading && (
            <div className="flex flex-col items-center justify-center h-48 gap-3">
              <div className="w-10 h-10 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin" />
              <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest animate-pulse">
                Feature engineering → XGBoost → SHAP…
              </p>
            </div>
          )}

          {!loading && error && (
            <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs space-y-1">
              <p className="font-black">Error</p>
              <p className="opacity-80">{error}</p>
              <p className="text-[9px] text-rose-400/50 pt-1">
                Make sure backend is running: <code className="bg-slate-900 px-1 rounded">cd icu_cdss/src && python api.py</code>
              </p>
            </div>
          )}

          {!loading && predictions.length > 0 && (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2 bg-slate-950/60 border border-white/5 rounded-2xl px-4 py-3">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                  {rowCount} row{rowCount !== 1 ? 's' : ''} · {predictions.length} result{predictions.length !== 1 ? 's' : ''}
                </p>
                <div className="flex gap-2 flex-wrap">
                  {(['CRITICAL', 'HIGH', 'MODERATE', 'LOW'] as const).map(lvl => {
                    const n = predictions.filter(p => p.risk_level === lvl).length;
                    return n ? (
                      <span key={lvl} className={`text-[9px] font-black px-2 py-0.5 rounded-lg bg-slate-800 ${RISK_TEXT[lvl]}`}>
                        {n} {lvl}
                      </span>
                    ) : null;
                  })}
                </div>
              </div>
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1 custom-scrollbar">
                {predictions.map((pred, i) => <PredictionCard key={`${pred.patient_id}-${i}`} pred={pred} />)}
              </div>
            </>
          )}

          {!loading && !error && predictions.length === 0 && (
            <div className="flex flex-col items-center justify-center h-48 gap-3 text-center">
              <span className="text-4xl opacity-20">🏥</span>
              <p className="text-[10px] font-black text-slate-600 uppercase tracking-widest">
                {mode === 'csv' ? 'Upload a CSV to see predictions' : 'Enter vitals and run the model'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PatientUpload;
