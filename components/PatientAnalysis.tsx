
import React, { useState, useEffect } from 'react';
import { Patient, XAIFactor } from '../types';
import { getClinicalSummary, getXAIExplanation } from '../services/icuApi';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip as ReTooltip, CartesianGrid } from 'recharts';

interface Props {
  patient: Patient;
}

const CustomXaiTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as XAIFactor;
    return (
      <div className="bg-[#020617] border border-white/10 p-3 rounded-xl shadow-2xl backdrop-blur-xl">
        <p className="text-[10px] font-black text-white uppercase mb-2 tracking-widest">{data.feature}</p>
        <div className="space-y-1.5">
          <div className="flex justify-between items-center gap-6">
            <span className="text-[9px] text-slate-500 font-bold uppercase">Contribution</span>
            <span className={`text-[10px] font-black ${data.contribution > 0 ? 'text-indigo-400' : 'text-rose-400'}`}>
              {data.contribution > 0 ? '+' : ''}{data.contribution.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between items-center gap-6">
            <span className="text-[9px] text-slate-500 font-bold uppercase">Impact</span>
            <span className={`px-1.5 py-0.5 rounded-md text-[8px] font-black uppercase border ${
              data.impact === 'High' ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20' : 
              data.impact === 'Medium' ? 'bg-sky-500/10 text-sky-400 border-sky-500/20' : 
              'bg-slate-500/10 text-slate-400 border-slate-500/20'
            }`}>
              {data.impact}
            </span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const PatientAnalysis: React.FC<Props> = ({ patient }) => {
  const [summaryData, setSummaryData] = useState<{ highlights: string[]; fullSummary: string }>({ highlights: [], fullSummary: '' });
  const [xaiData, setXaiData] = useState<XAIFactor[]>([]);
  const [loading, setLoading] = useState(true);
  const [isSummaryExpanded, setIsSummaryExpanded] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const sum = await getClinicalSummary(patient);
        const xai = await getXAIExplanation(patient);
        
        setSummaryData(sum);
        setXaiData(xai);
      } catch (err) {
        console.error("Dashboard analysis error:", err);
      } finally {
        setLoading(false);
        setIsSummaryExpanded(false);
      }
    };
    fetchData();
  }, [patient]);

  if (loading) {
    return (
      <div className="h-[500px] flex flex-col items-center justify-center p-12 bg-slate-900/20 rounded-[3rem] border border-white/5 shadow-2xl">
        <div className="w-12 h-12 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
        <p className="mt-8 font-black text-slate-500 animate-pulse tracking-widest uppercase text-[10px]">Analyzing Clinical Path...</p>
      </div>
    );
  }

  const healthMetrics = [
    { label: 'Heart Rate', val: patient.vitals.heartRate, unit: 'BPM', color: 'text-rose-400' },
    { label: 'Blood Pressure', val: patient.vitals.systolicBP, unit: 'mmHg', color: 'text-sky-400' },
    { label: 'Oxygen level', val: patient.vitals.oxygenSaturation, unit: '%', color: 'text-emerald-400' },
    { label: 'Infection (WBC)', val: patient.labs.wbc, unit: 'k/uL', color: 'text-purple-400' },
    { label: 'Kidney Filter', val: patient.labs.creatinine, unit: 'mg/dL', color: 'text-amber-400' },
    { label: 'Body Stress', val: patient.labs.lactate, unit: 'mmol/L', color: 'text-orange-400' },
  ];

  const paragraphs = summaryData.fullSummary.split('\n').filter(p => p.trim());
  const displayedParagraphs = isSummaryExpanded ? paragraphs : paragraphs.slice(0, 2);
  const hasMore = paragraphs.length > 2;

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 min-w-0">
      
      {/* 1. Dashboard Header: Patient Summary */}
      <div className="bg-slate-900/50 backdrop-blur-xl rounded-[2.5rem] p-6 md:p-8 border border-white/5 shadow-2xl">
        <div className="flex flex-col lg:flex-row gap-8 justify-between items-start">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-3">
              <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest border ${
                patient.acuityLevel === 'Critical' ? 'bg-red-500/20 text-red-400 border-red-500/30 shadow-lg shadow-red-500/10' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
              }`}>
                {patient.acuityLevel} Care
              </span>
              <span className="text-[10px] font-mono font-bold text-slate-600 bg-white/5 px-2 py-0.5 rounded uppercase">ID: {patient.id}</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-black text-white tracking-tighter mb-2 truncate">{patient.name}</h2>
            <p className="text-indigo-300 font-bold mb-6 text-lg">{patient.diagnosis}</p>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-red-500/[0.03] p-4 rounded-2xl border border-red-500/10">
                <p className="text-[9px] font-black text-red-400 uppercase mb-2 tracking-widest">Active Allergies</p>
                <div className="flex flex-wrap gap-2 text-xs font-bold text-red-100">
                  {patient.allergies.length > 0 ? patient.allergies.join(', ') : 'No known allergies reported.'}
                </div>
              </div>
              <div className="bg-indigo-500/[0.03] p-4 rounded-2xl border border-indigo-500/10">
                <p className="text-[9px] font-black text-indigo-400 uppercase mb-2 tracking-widest">Patient History</p>
                <div className="text-xs font-bold text-indigo-100 line-clamp-2">
                  {patient.preExistingConditions.join(', ')}
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-2 gap-3 w-full lg:w-auto">
            {healthMetrics.map(s => (
              <div key={s.label} className="bg-slate-950/60 p-4 rounded-2xl border border-white/5 hover:border-indigo-500/30 transition-all min-w-[130px]">
                <p className="text-[9px] font-black text-slate-500 uppercase mb-1 truncate">{s.label}</p>
                <p className={`text-xl font-black ${s.color} truncate`}>{s.val}<span className="text-[9px] font-normal text-slate-600 ml-1">{s.unit}</span></p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Reports Section */}
        <div className="lg:col-span-8 space-y-8 min-w-0">
          
          {/* Urgent Highlights Card */}
          <section className="bg-slate-900/40 rounded-[2rem] p-8 border border-white/5 shadow-2xl">
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 bg-amber-500/10 rounded-xl border border-amber-500/20">
                <svg className="w-6 h-6 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
              </div>
              <div>
                <h3 className="text-xl font-black text-white tracking-tight uppercase">Care Highlights</h3>
                <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Prioritized Health Observations</p>
              </div>
            </div>
            <div className="space-y-4">
              {summaryData.highlights.map((h, i) => (
                <div key={i} className="flex gap-4 items-start p-4 rounded-2xl bg-slate-950/40 border border-white/5 hover:border-amber-500/30 transition-all shadow-lg">
                  <div className="flex-shrink-0 w-6 h-6 bg-amber-500/10 rounded border border-amber-500/20 flex items-center justify-center text-amber-500 font-black text-[10px]">{i+1}</div>
                  <p className="text-slate-200 text-sm font-bold leading-relaxed">{h}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Full Case Summary Narrative with Toggle */}
          <section className="bg-slate-900/40 rounded-[2rem] p-8 border border-white/5 shadow-2xl">
             <div className="flex items-center gap-4 mb-6">
                <div className="p-3 bg-indigo-500/10 rounded-xl border border-indigo-500/20">
                  <svg className="w-6 h-6 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                </div>
                <div>
                  <h3 className="text-xl font-black text-white tracking-tight uppercase">Detailed Health Narrative</h3>
                  <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Comprehensive Medical Review</p>
                </div>
             </div>
             
             <div className="relative">
               <div className={`p-6 bg-slate-950/50 rounded-2xl border border-white/5 text-slate-300 text-sm leading-relaxed font-medium transition-all duration-500 overflow-hidden ${isSummaryExpanded ? 'max-h-[3000px]' : 'max-h-[220px]'}`}>
                 {displayedParagraphs.map((para, idx) => (
                   <p key={idx} className="mb-4 last:mb-0">{para}</p>
                 ))}
                 
                 {!isSummaryExpanded && hasMore && (
                   <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-slate-950/90 via-slate-950/40 to-transparent pointer-events-none rounded-b-2xl"></div>
                 )}
               </div>
               
               {hasMore && (
                 <button 
                   onClick={() => setIsSummaryExpanded(!isSummaryExpanded)}
                   className="mt-4 flex items-center gap-2 text-indigo-400 text-[10px] font-black uppercase tracking-widest hover:text-indigo-300 transition-colors mx-auto"
                 >
                   {isSummaryExpanded ? 'Read Less' : 'Read Full Narrative'}
                   <svg className={`w-3 h-3 transform transition-transform duration-300 ${isSummaryExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M19 9l-7 7-7-7"></path></svg>
                 </button>
               )}
             </div>
          </section>

          {/* Treatment Record Table */}
          <section className="bg-slate-900/40 rounded-[2rem] border border-white/5 shadow-2xl overflow-hidden">
             <div className="p-6 border-b border-white/5 bg-slate-900/20">
               <h3 className="text-xs font-black text-slate-400 uppercase tracking-[0.2em]">Treatment Plan & Schedule</h3>
             </div>
             <div className="overflow-x-auto min-w-full">
               <table className="w-full text-left text-xs font-bold min-w-[600px]">
                 <thead>
                   <tr className="bg-slate-950/80 border-b border-white/5 text-slate-600 font-black uppercase tracking-widest">
                     <th className="px-6 py-4">Medication Item</th>
                     <th className="px-6 py-4">Dose Detail</th>
                     <th className="px-6 py-4">Frequency</th>
                     <th className="px-6 py-4">Days</th>
                     <th className="px-6 py-4 text-right">Status</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-white/[0.03]">
                   {patient.medications.map((m, i) => (
                     <tr key={i} className="hover:bg-white/5 transition-colors">
                       <td className="px-6 py-5 text-slate-100">{m.name}<br/><span className="text-[8px] text-slate-600 uppercase mt-0.5">{m.route} Route</span></td>
                       <td className="px-6 py-5 text-slate-400">{m.dosage}</td>
                       <td className="px-6 py-5 text-slate-400">{m.frequency}</td>
                       <td className="px-6 py-5 text-indigo-300 font-mono font-bold">{m.durationDays}d</td>
                       <td className="px-6 py-5 text-right">
                         <span className={`px-2 py-0.5 rounded-full text-[8px] font-black uppercase border shadow-sm ${
                           m.status === 'Active' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 
                           m.status === 'Discontinued' ? 'bg-rose-500/10 text-rose-400 border-rose-500/30' :
                           'bg-slate-800 text-slate-500 border-white/5'
                         }`}>
                           {m.status}
                         </span>
                       </td>
                     </tr>
                   ))}
                 </tbody>
               </table>
             </div>
          </section>
        </div>

        {/* Intelligence Insight Column */}
        <div className="lg:col-span-4 space-y-8 min-w-0">
          
          {/* AI Decision Reasoning Card */}
          <section className="bg-slate-900/40 rounded-[2rem] p-8 border border-white/5 shadow-2xl flex flex-col min-h-[500px]">
            <h4 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.3em] mb-4">AI Clinical Reasoning (XAI)</h4>
            <p className="text-[10px] text-slate-500 mb-8 font-black leading-relaxed uppercase">
              Feature attribution for the <span className="text-white">{(patient.sepsisRisk * 100).toFixed(0)}% risk prediction</span>. Hover bars for details:
            </p>
            
            <div className="h-[320px] w-full relative">
              {xaiData && xaiData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={xaiData} layout="vertical" margin={{ left: -15, right: 30, top: 0, bottom: 0 }}>
                    <CartesianGrid horizontal={false} stroke="#1e293b" strokeDasharray="3 3" opacity={0.3} />
                    <XAxis type="number" hide />
                    <YAxis 
                      dataKey="feature" 
                      type="category" 
                      axisLine={false} 
                      tickLine={false} 
                      width={120} 
                      tick={{ fontSize: 9, fill: '#94a3b8', fontWeight: 800 }} 
                      tickFormatter={(val) => typeof val === 'string' ? val.split(' (')[0].toUpperCase() : val}
                    />
                    <ReTooltip 
                      cursor={{ fill: 'rgba(255,255,255,0.03)' }} 
                      content={<CustomXaiTooltip />}
                    />
                    <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                      {xaiData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.contribution > 0 ? '#6366f1' : '#f43f5e'} 
                          fillOpacity={0.9} 
                          className="transition-all duration-300 hover:fill-opacity-100"
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-[10px] font-black text-slate-600 uppercase italic">
                  No XAI Data Available
                </div>
              )}
            </div>

            <div className="mt-auto pt-8 flex justify-between items-center p-6 bg-slate-950/60 rounded-[1.5rem] border border-white/5 shadow-inner">
              <div className="text-center flex-1">
                <span className="block text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1">Confidence</span>
                <span className="text-lg font-black text-indigo-400">92%</span>
              </div>
              <div className="w-px h-8 bg-white/5 mx-4"></div>
              <div className="text-center flex-1">
                <span className="block text-[8px] font-black text-slate-600 uppercase tracking-widest mb-1">AI Engine</span>
                <span className="text-lg font-black text-emerald-400">ICU CDSS</span>
              </div>
            </div>
          </section>

          {/* Resource Simulation Prediction */}
          <section className="bg-gradient-to-br from-indigo-950/40 to-slate-950 p-8 rounded-[2rem] border border-indigo-500/20 shadow-2xl relative overflow-hidden group">
            <h4 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-6">Care Forecast</h4>
            <div className="space-y-6">
              <p className="text-lg text-slate-100 font-black leading-snug tracking-tight">
                Declining <span className="text-indigo-300">Kidney Filtration</span> indicates <span className="text-white underline decoration-indigo-500/40">Dialysis Gear</span> may be needed in <span className="text-indigo-400">~18h</span>.
              </p>
              <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden">
                <div className="h-full w-[80%] bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]"></div>
              </div>
              <button className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-black text-[10px] uppercase tracking-widest py-4 rounded-xl transition-all shadow-xl shadow-indigo-600/30 transform hover:-translate-y-0.5">
                Reserve Resource
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default PatientAnalysis;
