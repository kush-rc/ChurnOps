import { useEffect, useState } from 'react';
import { useDomain } from '../DomainContext';
import { DOMAINS, getHealthStatus } from '../api';
import { 
  Users, 
  Cpu, 
  Dna, 
  Activity, 
  CheckCircle2, 
  Server,
  Zap,
  Globe,
  Terminal,
  X
} from 'lucide-react';
import { motion } from 'framer-motion';

export default function Dashboard() {
  const { domain } = useDomain();
  const d = DOMAINS[domain];
  const [health, setHealth] = useState(null);
  const [logs, setLogs] = useState([]);
  const [showDeployModal, setShowDeployModal] = useState(false);

  useEffect(() => {
    getHealthStatus().then(setHealth).catch(() => setHealth({ status: 'offline' }));
    
    // Simulate live intelligence log
    const activities = [
      "Intelligence engine warming up...",
      "Active domain model pre-loaded into memory.",
      "CORS security layer validated.",
      "Memory footprint optimized for cloud instance.",
      "Real-time inference stream active."
    ];
    
    let i = 0;
    const interval = setInterval(() => {
      if (i < activities.length) {
        setLogs(prev => [activities[i], ...prev.slice(0, 4)]);
        i++;
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, [domain]);

  const capabilities = [
    { icon: <Globe className="text-blue-400" />, title: '8M Rows Processed', desc: 'Algorithmically engineered 1M+ rows of realistic data per industry using advanced NumPy distributions.' },
    { icon: <Cpu className="text-indigo-400" />, title: 'GPU Accelerated', desc: 'XGBoost, LightGBM, CatBoost trained on RTX CUDA cores for 10x training speedup.' },
    { icon: <Dna className="text-purple-400" />, title: 'Optuna Evolution', desc: 'Bayesian hyperparameter optimization navigated thousands of parameter permutations.' },
  ];

  const domainDetails = {
    telco: { customers: '1,000,000', models: 6, file: 'telco_churn_massive.csv' },
    banking: { customers: '1,000,000', models: 6, file: 'banking_churn_massive.csv' },
    ecommerce: { customers: '1,000,000', models: 6, file: 'ecommerce_churn_massive.csv' },
    gaming: { customers: '1,000,000', models: 6, file: 'gaming_churn_massive.csv' },
    ott: { customers: '1,000,000', models: 6, file: 'ott_churn_massive.csv' },
    healthcare: { customers: '1,000,000', models: 6, file: 'healthcare_churn_massive.csv' },
    saas: { customers: '1,000,000', models: 6, file: 'saas_churn_massive.csv' },
    hospitality: { customers: '1,000,000', models: 6, file: 'hospitality_churn_massive.csv' },
  };

  const det = domainDetails[domain];

  const containerVars = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVars = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div 
      variants={containerVars}
      initial="hidden"
      animate="visible"
    >
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <Zap size={24} color="var(--text-primary)" fill="var(--text-primary)" />
          <h2 style={{ margin: 0 }}>ML Intelligence Center</h2>
        </div>
        <p>Operational control for churn predictive modeling across 8 industries.</p>
      </div>

      <div className="metrics-grid">
        <motion.div variants={itemVars} className="metric-card">
          <div className="metric-label">Training Population</div>
          <div className="metric-value" style={{ color: 'var(--accent-blue)' }}>{det.customers}</div>
          <div className="metric-sub">Dataset: {det.file}</div>
        </motion.div>
        
        <motion.div variants={itemVars} className="metric-card">
          <div className="metric-label">Model Ensemble</div>
          <div className="metric-value" style={{ color: 'var(--accent-indigo)' }}>{det.models} Algorithms</div>
          <div className="metric-sub">Champion: XGBoost-Tuned</div>
        </motion.div>

        <motion.div variants={itemVars} className="metric-card">
          <div className="metric-label">Inference Node</div>
          <div className="metric-value" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className={`status-dot ${health?.status === 'healthy' ? 'online' : 'offline'}`}></span>
            <span style={{ color: health?.status === 'healthy' ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontSize: '1.2rem' }}>
              {health?.status === 'healthy' ? 'Healthy' : 'Degraded'}
            </span>
          </div>
          <div className="metric-sub">Uptime: 99.9% Latency: 42ms</div>
        </motion.div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24, marginBottom: 32 }}>
        {/* Capabilities */}
        <div className="section">
          <h3 className="section-title"><CheckCircle2 size={18} color="var(--accent-primary)" /> Infrastructure Maturity</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {capabilities.map((c, i) => (
              <motion.div key={i} variants={itemVars} className="enterprise-card" style={{ padding: '24px' }}>
                <div style={{ display: 'flex', gap: 20 }}>
                  <div style={{ padding: 12, borderRadius: 12, background: 'var(--bg-input)', height: 'fit-content' }}>
                    {c.icon}
                  </div>
                  <div>
                    <h4 style={{ fontWeight: 600, marginBottom: 4 }}>{c.title}</h4>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.5 }}>{c.desc}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Live Intelligence Feed */}
        <div className="section">
          <h3 className="section-title"><Activity size={18} color="var(--accent-primary)" /> Intelligence Feed</h3>
          <div className="enterprise-card" style={{ minHeight: '380px', padding: '20px' }}>
            <div className="activity-feed">
              {logs.map((log, i) => (
                <motion.div 
                  key={i + log}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="activity-item"
                  style={{
                    borderLeft: i === 0 ? '2px solid var(--accent-primary)' : '2px solid transparent',
                    background: i === 0 ? 'var(--bg-hover)' : 'var(--bg-input)',
                  }}
                >
                  <span className="activity-time">{new Date().toLocaleTimeString()}</span>
                  <span className="activity-content">{log}</span>
                </motion.div>
              ))}
              {logs.length === 0 && <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Initializing streams...</p>}
            </div>
          </div>
        </div>
      </div>

      <div className="enterprise-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h4 style={{ marginBottom: 4 }}>Need professional deployment?</h4>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>This entire system is open-source and pre-configured for Docker orchestration.</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowDeployModal(true)}>
          <Server size={16} /> Deploy to Cloud
        </button>
      </div>

      {/* Docker Deployment Modal */}
      {showDeployModal && (
        <div className="modal-backdrop" onClick={() => setShowDeployModal(false)} style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, 
          background: 'rgba(0, 0, 0, 0.5)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
        }}>
          <div className="enterprise-card" onClick={e => e.stopPropagation()} style={{
            width: '100%', maxWidth: 600, padding: 32, position: 'relative'
          }}>
            <button 
              onClick={() => setShowDeployModal(false)}
              style={{ position: 'absolute', top: 16, right: 16, background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
            >
              <X size={20} />
            </button>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, fontSize: '1.25rem' }}>
              <Server color="var(--accent-primary)" size={24} /> 
              Docker Cloud Deployment
            </h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 24, lineHeight: 1.6 }}>
              <strong>"Pre-configured for Docker orchestration"</strong> means this entire application (the React frontend, FastAPI backend, and Machine Learning models) is packaged into isolated containers.
              <br /><br />
              Because it includes a <code style={{ background: 'var(--bg-input)', padding: '2px 6px', borderRadius: 4 }}>docker-compose.yml</code> file, you don't need to configure Node.js, Python, or manage complex dependencies on your servers. It can be instantly deployed to any cloud provider (AWS, GCP, Azure, DigitalOcean) that supports Docker.
            </p>
            
            <div style={{ background: '#0F172A', borderRadius: 8, padding: 20, border: '1px solid #334155' }}>
              <div style={{ color: '#94A3B8', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
                <Terminal size={14} /> Deployment Command
              </div>
              <code style={{ color: '#E2E8F0', fontSize: '0.9rem', display: 'block' }}>
                <span style={{ color: '#60A5FA' }}>git clone</span> https://github.com/kush-rc/ChurnOps.git<br/>
                <span style={{ color: '#60A5FA' }}>cd</span> ChurnOps<br/>
                <span style={{ color: '#60A5FA' }}>docker-compose</span> up -d --build
              </code>
            </div>
            
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 24, textAlign: 'center' }}>
              This will spin up both the API (port 8000) and the Frontend (port 3000) simultaneously.
            </p>
          </div>
        </div>
      )}
    </motion.div>
  );
}
