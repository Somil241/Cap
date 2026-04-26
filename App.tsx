import React, { useEffect, useRef, useState } from 'react';
import { MOCK_FORECAST, MOCK_RESOURCES } from './constants';
import { BedDemandData, Patient, ResourceNeed } from './types';
import PatientCard from './components/PatientCard';
import PatientAnalysis from './components/PatientAnalysis';
import PredictiveChart from './components/PredictiveChart';
import ResourceForecastCard from './components/ResourceForecastCard';
import PatientUpload from './components/PatientUpload';
import {
  fetchBedForecast,
  fetchPatients,
  fetchResources,
} from './services/icuApi';

const App: React.FC = () => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [forecast, setForecast] = useState<BedDemandData[]>(MOCK_FORECAST);
  const [resources, setResources] = useState<ResourceNeed[]>(MOCK_RESOURCES);
  const [isProcessing, setIsProcessing] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDashboard = async (limit = 12) => {
    setIsProcessing(true);
    setBootError(null);
    try {
      const [pts, fc, res] = await Promise.all([
        fetchPatients(limit),
        fetchBedForecast(undefined, 7),
        fetchResources(),
      ]);
      setPatients(pts);
      setSelectedPatient(prev => {
        if (!pts.length) return null;
        const stillThere = prev && pts.find(p => p.id === prev.id);
        return stillThere ?? pts[0];
      });
      if (fc.length) setForecast(fc);
      if (res.length) setResources(res);
    } catch (err: any) {
      console.error('CDSS API error:', err);
      setBootError(err?.message ?? String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => { loadDashboard(12); }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await loadDashboard(12);
    if (event.target) event.target.value = '';
  };

  const criticalCount = patients.filter(p => p.sepsisRisk > 0.7).length;
  const avgLOS = patients.length
    ? (patients.reduce((acc, p) => acc + p.predictedLOS, 0) / patients.length).toFixed(1)
    : '—';

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 overflow-x-hidden">
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-500/5 blur-[120px] rounded-full"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[120px] rounded-full"></div>
      </div>

      <header className="bg-slate-950/60 backdrop-blur-2xl border-b border-white/5 px-4 md:px-8 py-4 sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto flex justify-between items-center gap-4">
          <div className="flex items-center gap-4">
            <div className="bg-gradient-to-tr from-indigo-600 to-blue-500 p-2 rounded-xl shadow-lg shadow-indigo-500/20">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-black text-white tracking-tight leading-none">SmartCare <span className="text-indigo-400">ICU</span></h1>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mt-1">Hospital Intelligence</p>
            </div>
          </div>

          <div className="flex gap-4 items-center">
            <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-black uppercase tracking-widest px-4 py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-600/20 whitespace-nowrap disabled:opacity-60"
            >
              {isProcessing ? 'Loading…' : 'Refresh Cohort'}
            </button>
            <div className="hidden sm:flex items-center gap-3 bg-white/5 border border-white/10 px-3 py-1.5 rounded-xl">
              <div className="text-right">
                <p className="text-[8px] font-black text-slate-500 uppercase tracking-tighter">Attending</p>
                <p className="text-[11px] font-bold text-indigo-300">Dr. Malhotra</p>
              </div>
              <img className="w-8 h-8 rounded-lg" src="https://api.dicebear.com/7.x/avataaars/svg?seed=Rajesh" alt="Dr" />
            </div>
          </div>
        </div>
      </header>

      {bootError && (
        <div className="max-w-[1600px] mx-auto w-full px-4 md:px-8 mt-4">
          <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-bold">
            CDSS backend unreachable: {bootError}. Start the API with
            <code className="mx-1 px-1.5 py-0.5 bg-slate-900 rounded">cd icu_cdss/src && ../.venv/bin/python api.py</code>
            and refresh.
          </div>
        </div>
      )}

      <main className="max-w-[1600px] mx-auto w-full flex-1 p-4 md:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

        {/* Side Panel */}
        <aside className="lg:col-span-3 space-y-6">
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-4">
            <div className="bg-slate-900/40 p-5 rounded-3xl border border-white/5 shadow-xl">
              <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Critical Patients</p>
              <div className="text-3xl font-black text-white">{criticalCount}</div>
              <div className="mt-3 w-full bg-slate-800 rounded-full h-1 overflow-hidden">
                <div className="bg-red-500 h-full" style={{ width: `${Math.min(100, criticalCount * 12)}%` }}></div>
              </div>
            </div>
            <div className="bg-slate-900/40 p-5 rounded-3xl border border-white/5 shadow-xl">
              <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Average Recovery</p>
              <div className="text-3xl font-black text-white">{avgLOS} <span className="text-xs font-medium text-slate-500">Days</span></div>
              <div className="mt-3 w-full bg-slate-800 rounded-full h-1 overflow-hidden">
                <div className="bg-indigo-500 h-full w-[60%]"></div>
              </div>
            </div>
          </div>

          <div className="bg-slate-900/30 rounded-[2rem] border border-white/5 flex flex-col h-[500px] shadow-2xl overflow-hidden">
            <div className="p-5 border-b border-white/5 bg-slate-900/60">
              <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest">Active Patient List</h3>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
              {isProcessing && patients.length === 0 && (
                <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest text-center mt-12">Loading patients…</div>
              )}
              {!isProcessing && patients.length === 0 && !bootError && (
                <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest text-center mt-12">No patients available</div>
              )}
              {patients.map(p => (
                <PatientCard
                  key={p.id}
                  patient={p}
                  isActive={selectedPatient?.id === p.id}
                  onClick={() => setSelectedPatient(p)}
                />
              ))}
            </div>
          </div>

          <div className="bg-slate-900/40 p-6 rounded-[2rem] border border-white/5 shadow-xl">
            <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] mb-6">Equipment Status</h3>
            <div className="space-y-5">
              {resources.map(res => {
                const inUse = Math.round(res.current * res.utilizationRate);
                return (
                  <div key={res.id} className="group/equip relative">
                    <div className="flex justify-between text-[10px] font-bold mb-1.5">
                      <span className="text-slate-300">{res.type}</span>
                      <span className={res.status === 'Critical' ? 'text-red-400' : res.status === 'Warning' ? 'text-amber-400' : 'text-emerald-400'}>{res.status}</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${res.status === 'Critical' ? 'bg-red-500' : res.status === 'Warning' ? 'bg-amber-500' : 'bg-emerald-500'}`}
                        style={{ width: `${Math.min(100, res.utilizationRate * 100)}%` }}
                      />
                    </div>
                    <div className="absolute left-0 top-full mt-2 px-3 py-2.5 bg-slate-800 border border-white/10 rounded-xl shadow-xl opacity-0 invisible group-hover/equip:opacity-100 group-hover/equip:visible transition-all duration-200 z-50 pointer-events-none min-w-[200px]">
                      <div className="text-[10px] font-black text-slate-400 uppercase tracking-wider mb-2">{res.type}</div>
                      <div className="space-y-1.5 text-[11px]">
                        <div className="flex justify-between gap-4">
                          <span className="text-slate-500">In use:</span>
                          <span className="font-bold text-slate-200">{inUse}</span>
                        </div>
                        <div className="flex justify-between gap-4">
                          <span className="text-slate-500">Predicted required:</span>
                          <span className="font-bold text-indigo-300">{res.predicted}</span>
                        </div>
                      </div>
                      <div className="absolute -top-1 left-4 w-2 h-2 bg-slate-800 border-l border-t border-white/10 rotate-45" />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <div className="lg:col-span-9 space-y-8 min-w-0">
          {selectedPatient ? (
            <PatientAnalysis patient={selectedPatient} />
          ) : (
            <div className="h-[500px] flex flex-col items-center justify-center p-12 bg-slate-900/20 rounded-[3rem] border border-white/5 shadow-2xl">
              <div className="w-12 h-12 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
              <p className="mt-8 font-black text-slate-500 animate-pulse tracking-widest uppercase text-[10px]">
                {isProcessing ? 'Loading ICU dashboard…' : 'Select a patient'}
              </p>
            </div>
          )}

          <div className="bg-slate-900/20 p-8 rounded-[3rem] border border-white/5 space-y-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <h2 className="text-2xl font-black text-white tracking-tight">ICU Occupancy Trends</h2>
              <div className="flex items-center gap-3 bg-slate-950/60 px-4 py-2 rounded-2xl border border-white/5">
                <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'}`}></div>
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{isProcessing ? 'Syncing…' : 'Real-time'}</span>
              </div>
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-10">
              <div className="xl:col-span-7 overflow-hidden">
                <PredictiveChart data={forecast} />
              </div>
              <div className="xl:col-span-5 h-full">
                <ResourceForecastCard />
              </div>
            </div>
          </div>

          {/* ── NEW: Patient Upload Panel ── */}
          <PatientUpload />

        </div>
      </main>

      <footer className="mt-auto py-8 px-8 border-t border-white/5 text-center text-[9px] font-black text-slate-600 uppercase tracking-[0.3em]">
        ICU Intelligence Unit • MIMIC-IV CDSS Core • HIPAA Secure
      </footer>
    </div>
  );
};

export default App;
