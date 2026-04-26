
import React from 'react';

const ResourceForecastCard: React.FC = () => {
  return (
    <div className="relative group lg:h-full">
      {/* Outer border glow effect */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-[3rem] blur opacity-75 group-hover:opacity-100 transition duration-1000 group-hover:duration-200"></div>
      
      <div className="relative bg-[#0b0e1f] border border-white/10 rounded-[2.5rem] overflow-hidden flex flex-col h-full shadow-2xl">
        {/* Header Section */}
        <div className="flex justify-between items-center px-8 py-5 border-b border-white/5 bg-white/[0.02]">
          <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.3em] drop-shadow-sm">Care Resource Forecast</h3>
          <div className="bg-slate-900/80 border border-white/10 rounded-full px-3 py-1.5 flex items-center gap-2.5 shadow-inner">
             <div className="flex flex-col">
               <span className="text-[7px] font-black text-slate-500 uppercase tracking-widest leading-none mb-0.5">Simulator Status</span>
               <span className="text-[8px] font-black text-emerald-400 uppercase tracking-widest leading-none">Operational</span>
             </div>
             <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)] animate-pulse"></div>
          </div>
        </div>

        {/* Content Section */}
        <div className="p-8 md:p-10 flex-1 flex flex-col justify-between space-y-8">
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_12px_rgba(249,115,22,0.8)] animate-pulse"></div>
              <h4 className="text-[11px] font-black text-orange-400 uppercase tracking-[0.3em]">Equipment Alerts</h4>
            </div>

            <div className="relative">
              <p className="text-3xl md:text-4xl font-bold text-white leading-[1.15] tracking-tight">
                Prediction model indicates a <br/>
                <span className="text-indigo-400 font-black inline-block mt-2 drop-shadow-[0_0_15px_rgba(129,140,248,0.2)]">24% surge</span> <br/>
                <span className="text-slate-100">in ventilator demand within the </span> 
                <span className="relative inline-block text-white ml-1">
                  next 48 hours.
                  <div className="absolute -bottom-2 left-0 right-0 h-0.5 bg-slate-700/60 rounded-full"></div>
                </span>
              </p>
            </div>
          </div>

          {/* Recommendation Box */}
          <div className="mt-4 p-6 bg-gradient-to-br from-indigo-600/20 to-purple-600/10 border border-indigo-500/30 rounded-[2.2rem] flex items-start gap-5 relative overflow-hidden group/box shadow-lg">
            <div className="absolute inset-0 bg-indigo-500/[0.03] backdrop-blur-sm"></div>
            
            <div className="relative z-10 flex-shrink-0 w-14 h-14 bg-gradient-to-br from-indigo-500 to-indigo-700 rounded-2xl flex items-center justify-center shadow-2xl shadow-indigo-950 ring-1 ring-white/20">
              <svg className="w-7 h-7 text-white drop-shadow-md" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
              </svg>
            </div>
            
            <div className="relative z-10 py-1">
              <h5 className="text-[10px] font-black text-white uppercase tracking-[0.2em] mb-2.5 opacity-90">System Recommendation</h5>
              <p className="text-[13px] text-slate-300 font-bold leading-relaxed tracking-wide">
                Consider transferring stable patients to step-down wards to free up high-acuity beds.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResourceForecastCard;
