
import React from 'react';
import { Patient } from '../types';

interface Props {
  patient: Patient;
  isActive: boolean;
  onClick: () => void;
}

const PatientCard: React.FC<Props> = ({ patient, isActive, onClick }) => {
  // Tuned to the trained model's calibration. Because the sepsis classifier
  // is fit on a SMOTE-balanced train set, positive probabilities cluster
  // high; the F1-optimal threshold on raw-prevalence val data is ~0.73.
  // Backend writes models/sepsis_threshold.json; mirror those values here.
  const getRiskStyles = (risk: number) => {
    if (risk >= 0.73) return 'text-red-400 bg-red-950/30 border-red-900/50';
    if (risk >= 0.44) return 'text-orange-400 bg-orange-950/30 border-orange-900/50';
    return 'text-emerald-400 bg-emerald-950/30 border-emerald-900/50';
  };

  const getAcuityColor = (acuity: Patient['acuityLevel']) => {
    switch(acuity) {
      case 'Critical': return 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]';
      case 'High': return 'bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]';
      case 'Moderate': return 'bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.6)]';
      case 'Low': return 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]';
    }
  };

  return (
    <div 
      onClick={onClick}
      className={`p-4 rounded-2xl border transition-all duration-300 group ${
        isActive 
          ? 'bg-slate-800/80 border-indigo-500/50 shadow-[0_0_20px_rgba(99,102,241,0.15)] ring-1 ring-indigo-500/20' 
          : 'bg-slate-900/40 border-slate-800/60 hover:border-slate-700 hover:bg-slate-800/40 cursor-pointer'
      }`}
    >
      <div className="flex justify-between items-start mb-4">
        <div>
          <h4 className={`font-semibold transition-colors ${isActive ? 'text-indigo-300' : 'text-slate-200 group-hover:text-slate-100'}`}>
            {patient.name}
          </h4>
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-tight">
            ID:{patient.id} • {patient.age}y • {patient.gender}
          </span>
        </div>
        <div className={`w-2.5 h-2.5 rounded-full ${getAcuityColor(patient.acuityLevel)}`} />
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className={`py-2 px-1 rounded-xl border text-center ${getRiskStyles(patient.sepsisRisk)}`}>
          <div className="text-[9px] uppercase font-bold tracking-tighter opacity-60 font-mono">SEPSIS_PROB</div>
          <div className="text-lg font-black">{(patient.sepsisRisk * 100).toFixed(0)}%</div>
        </div>
        <div className="py-2 px-1 rounded-xl border border-slate-800 bg-slate-950/40 text-slate-300 text-center">
          <div className="text-[9px] uppercase font-bold tracking-tighter opacity-40 font-mono">EST_LOS</div>
          <div className="text-lg font-black">{patient.predictedLOS}<span className="text-[10px] ml-0.5 opacity-50">d</span></div>
        </div>
      </div>

      <div className="flex justify-between items-center text-[10px] text-slate-500 font-mono">
        <div className="flex gap-2">
          <span>GCS:{patient.clinicalScores.gcs}</span>
          <span>SOFA:{patient.clinicalScores.sofa}</span>
        </div>
        <div className="truncate italic max-w-[100px] text-right">{patient.diagnosis}</div>
      </div>
    </div>
  );
};

export default PatientCard;
