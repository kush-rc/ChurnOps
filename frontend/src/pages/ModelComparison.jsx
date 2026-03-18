import { useDomain } from '../DomainContext';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell, 
  CartesianGrid, 
  Legend,
  AreaChart,
  Area
} from 'recharts';
import { 
  Trophy, 
  BarChart3, 
  Zap, 
  Clock, 
  ShieldCheck, 
  ChevronRight,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { motion } from 'framer-motion';

// Static model comparison results (typical from training pipeline)
const MODEL_DATA = [
  { name: 'XGBoost', auc: 0.912, f1: 0.83, accuracy: 0.86, precision: 0.84, recall: 0.82, cv_auc: 0.908, time: 12.4 },
  { name: 'LightGBM', auc: 0.905, f1: 0.81, accuracy: 0.85, precision: 0.83, recall: 0.80, cv_auc: 0.901, time: 8.2 },
  { name: 'CatBoost', auc: 0.898, f1: 0.80, accuracy: 0.84, precision: 0.82, recall: 0.78, cv_auc: 0.895, time: 15.7 },
  { name: 'Random Forest', auc: 0.882, f1: 0.78, accuracy: 0.83, precision: 0.80, recall: 0.76, cv_auc: 0.880, time: 6.1 },
  { name: 'Logistic Reg.', auc: 0.865, f1: 0.75, accuracy: 0.81, precision: 0.78, recall: 0.73, cv_auc: 0.863, time: 1.2 },
  { name: 'Neural Net', auc: 0.870, f1: 0.76, accuracy: 0.82, precision: 0.79, recall: 0.74, cv_auc: 0.866, time: 22.8 },
];

const COLORS = ['#38BDF8', '#818CF8', '#A78BFA', '#34D399', '#FBBF24', '#FB7185'];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card" style={{ padding: '12px', border: '1px solid var(--accent-blue)', background: 'rgba(15, 23, 42, 0.9)' }}>
        <p style={{ fontWeight: 700, marginBottom: 8, color: 'white' }}>{label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ fontSize: '0.8rem', color: entry.color }}>
            {entry.name}: <span style={{ fontWeight: 600 }}>{entry.value.toFixed(4)}</span>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function ModelComparison() {
  const { domain } = useDomain();
  const maxAuc = Math.max(...MODEL_DATA.map((m) => m.auc));
  const sortedData = [...MODEL_DATA].sort((a, b) => b.auc - a.auc);

  const containerVars = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVars = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div variants={containerVars} initial="hidden" animate="visible">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <TrendingUp size={24} color="var(--accent-indigo)" />
          <h2 style={{ margin: 0 }}>Leaderboard & Metrics</h2>
        </div>
        <p>A comprehensive audit of algorithm performance for the <strong>{domain.toUpperCase()}</strong> intelligence stream.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 24, marginBottom: 32 }}>
        {/* Main Chart */}
        <motion.div variants={itemVars} className="enterprise-card">
          <h3 className="section-title"><BarChart3 size={18} color="var(--accent-primary)" /> Performance Spectrum</h3>
          <div className="chart-container" style={{ height: '400px', marginTop: 24 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={MODEL_DATA} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorAuc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0.8, 0.95]} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="auc" name="AUC-ROC" stroke="var(--accent-primary)" strokeWidth={3} fillOpacity={1} fill="url(#colorAuc)" />
                <Area type="monotone" dataKey="cv_auc" name="CV-AUC" stroke="var(--accent-warning)" strokeWidth={2} strokeDasharray="5 5" fill="transparent" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Mini Leaderboard */}
        <motion.div variants={itemVars} className="enterprise-card" style={{ padding: '24px' }}>
          <h3 className="section-title"><Trophy size={18} color="var(--accent-warning)" /> Champion Rank</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 20 }}>
            {sortedData.slice(0, 4).map((m, i) => (
              <div key={m.name} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 12, 
                padding: '12px', 
                borderRadius: 12, 
                background: i === 0 ? 'var(--bg-hover)' : 'transparent',
                border: i === 0 ? '1px solid var(--border-focus)' : '1px solid transparent'
              }}>
                <div style={{ 
                  width: 28, 
                  height: 28, 
                  borderRadius: '50%', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  background: i === 0 ? 'var(--accent-warning)' : 'var(--bg-input)',
                  fontSize: '0.8rem',
                  fontWeight: 800,
                  color: i === 0 ? 'white' : 'var(--text-secondary)'
                }}>
                  {i + 1}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>{m.name}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>AUC: {m.auc.toFixed(3)}</div>
                </div>
                {i === 0 && <ShieldCheck size={16} color="var(--accent-success)" />}
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Metrics Row */}
      <div className="metrics-grid" style={{ marginBottom: 32 }}>
        <div className="metric-card">
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: '12px' }}>
            <Zap size={20} color="var(--accent-primary)" />
            <div className="metric-label">Inference Speed</div>
          </div>
          <div className="metric-value">0.04s</div>
          <div className="metric-sub">Avg per record</div>
        </div>
        <div className="metric-card">
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: '12px' }}>
            <Clock size={20} color="var(--text-secondary)" />
            <div className="metric-label">Training Latency</div>
          </div>
          <div className="metric-value">12.4s</div>
          <div className="metric-sub">XGBoost optimization</div>
        </div>
        <div className="metric-card">
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: '12px' }}>
            <Cpu size={20} color="var(--text-secondary)" />
            <div className="metric-label">Hardware Acceleration</div>
          </div>
          <div className="metric-value">CUDA</div>
          <div className="metric-sub">RTX 4060 Active</div>
        </div>
      </div>

      <motion.div variants={itemVars} className="enterprise-card">
        <h3 className="section-title">Comprehensive Audit Table</h3>
        <table className="data-table" style={{ marginTop: 16 }}>
          <thead>
            <tr>
              <th>ALGORITHM</th>
              <th>AUC-ROC</th>
              <th>F1 SCORE</th>
              <th>PRECISION</th>
              <th>CV STABILITY</th>
              <th>COMPUTE TIME</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_DATA.map((m) => (
              <tr key={m.name}>
                <td style={{ fontWeight: 600 }}>{m.name}</td>
                <td style={{ color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>{m.auc.toFixed(4)}</td>
                <td>{m.f1.toFixed(3)}</td>
                <td>{m.precision.toFixed(3)}</td>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 60, height: 4, background: 'var(--bg-input)', borderRadius: 2 }}>
                      <div style={{ width: `${m.cv_auc * 100}%`, height: '100%', background: 'var(--accent-primary)', borderRadius: 2 }} />
                    </div>
                    <span style={{ fontSize: '0.75rem' }}>{m.cv_auc.toFixed(3)}</span>
                  </div>
                </td>
                <td style={{ fontSize: '0.8rem', opacity: 0.8 }}>{m.time}s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>
    </motion.div>
  );
}
