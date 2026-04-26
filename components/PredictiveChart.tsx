
import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { BedDemandData } from '../types';

interface Props {
  data: BedDemandData[];
}

const PredictiveChart: React.FC<Props> = ({ data }) => {
  return (
    <div className="h-[320px] w-full min-h-[320px] relative overflow-visible">
      <div className="flex flex-wrap justify-between items-center mb-6 px-2 gap-2">
        <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Patient Volume Forecast</h3>
        <div className="flex gap-4">
          <span className="flex items-center gap-1.5 text-[9px] font-black text-slate-600 uppercase tracking-widest">
            <span className="w-2 h-2 rounded-full bg-indigo-500 shadow-sm shadow-indigo-500/40"></span> Demand
          </span>
          <span className="flex items-center gap-1.5 text-[9px] font-black text-slate-600 uppercase tracking-widest">
            <span className="w-2 h-2 rounded-full bg-red-500 shadow-sm shadow-red-500/40"></span> Capacity
          </span>
        </div>
      </div>
      <div className="w-full h-[calc(100%-40px)] min-h-[260px]">
        {data && data.length > 0 ? (
          <ResponsiveContainer width="99%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
              <defs>
                <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b" opacity={0.3} />
              <XAxis 
                dataKey="date" 
                axisLine={false} 
                tickLine={false} 
                tick={{ fontSize: 9, fill: '#64748b', fontWeight: 900 }} 
                dy={10}
              />
              <YAxis 
                axisLine={false} 
                tickLine={false} 
                tick={{ fontSize: 9, fill: '#64748b', fontWeight: 900 }} 
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0f172a', 
                  borderRadius: '12px', 
                  border: '1px solid #334155', 
                  padding: '10px 14px',
                  fontSize: '11px',
                  fontWeight: '900',
                  boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.4)'
                }}
                itemStyle={{ color: '#818cf8', padding: '0' }}
                labelStyle={{ color: '#f8fafc', marginBottom: '2px', fontWeight: '900' }}
                cursor={{ stroke: '#334155', strokeWidth: 1 }}
              />
              <ReferenceLine y={25} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.5} strokeWidth={2} />
              <Area 
                type="monotone" 
                dataKey="predictedDemand" 
                stroke="#818cf8" 
                strokeWidth={3}
                fillOpacity={1} 
                fill="url(#colorDemand)" 
                name="Estimated Count"
                animationDuration={1500}
                isAnimationActive={true}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-slate-700 font-black text-xs uppercase italic">
            Trend Data Unavailable
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictiveChart;
