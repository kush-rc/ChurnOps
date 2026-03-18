import { useEffect, useState } from 'react';
import { useDomain } from '../DomainContext';
import { explainPrediction, DOMAINS, DOMAIN_FIELDS } from '../api';
import { 
  Binary, 
  Search, 
  Activity, 
  TrendingUp, 
  Info, 
  ShieldAlert, 
  BrainCircuit,
  Maximize2,
  ListFilter
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Explainability() {
  const { domain } = useDomain();
  const d = DOMAINS[domain];
  const [importances, setImportances] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    fetchExplanation();
  }, [domain]);

  async function fetchExplanation() {
    setLoading(true);
    try {
      const config = DOMAIN_FIELDS[domain];
      const defaults = {};
      config.fields.forEach((f) => {
        if (f.type === 'select') defaults[f.name] = f.options[0];
        else defaults[f.name] = f.default ?? 0;
      });

      const result = await explainPrediction(domain, defaults);
      setPrediction(result.prediction);
      setImportances(result.feature_importances || []);
    } catch (err) {
      setPrediction(null);
      setImportances(null);
    } finally {
      setLoading(false);
    }
  }

  const maxImportance = importances ? Math.max(...importances.map((fi) => Math.abs(fi.shap_value)), 0.01) : 1;

  const containerVars = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVars = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 }
  };

  return (
    <motion.div variants={containerVars} initial="hidden" animate="visible">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <Binary size={24} color="var(--accent-purple)" />
          <h2 style={{ margin: 0 }}>Interpretability Engine</h2>
        </div>
        <p>A deep audit of model decision logic through game-theoretic feature contribution analysis.</p>
      </div>

      <div className="enterprise-card" style={{ marginBottom: 32, borderLeft: '4px solid var(--accent-primary)', background: 'var(--bg-input)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Info size={16} color="var(--accent-primary)" />
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
            Visualizing <strong>Local Interpretability</strong> for a typical customer profile in the <strong>{d.label}</strong> domain.
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 24 }}>
        {/* Prediction Summary Sidebar */}
        <motion.div variants={itemVars} className="enterprise-card" style={{ height: 'fit-content' }}>
          <h3 className="section-title"><Activity size={18} color="var(--accent-primary)" /> State Audit</h3>
          {loading ? (
            <div style={{ padding: '40px 0', textAlign: 'center' }}>
              <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }} style={{ display: 'inline-block' }}>
                <BrainCircuit size={32} color="var(--text-muted)" />
              </motion.div>
            </div>
          ) : prediction ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 20 }}>
              <div className="metric-card" style={{ padding: '16px' }}>
                <div className="metric-label">Churn Probability</div>
                <div className="metric-value" style={{ fontSize: '2rem', color: prediction.churn_probability > 0.5 ? 'var(--accent-danger)' : 'var(--accent-success)' }}>
                  {(prediction.churn_probability * 100).toFixed(1)}%
                </div>
              </div>
              <div className="metric-card" style={{ padding: '16px' }}>
                <div className="metric-label">Model Categorization</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8 }}>
                  {prediction.label === 'Churned' ? <ShieldAlert size={18} color="var(--accent-danger)" /> : <Search size={18} color="var(--accent-success)" />}
                  <span style={{ fontWeight: 700, fontSize: '0.9rem', color: prediction.label === 'Churned' ? 'var(--accent-danger)' : 'var(--accent-success)' }}>
                    {prediction.label.toUpperCase()}
                  </span>
                </div>
              </div>
              <div className="metric-card" style={{ padding: '16px' }}>
                <div className="metric-label">Determinism Confidence</div>
                <div className="metric-value" style={{ fontSize: '1.25rem', color: 'var(--text-primary)' }}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Awaiting engine stream...</p>
          )}
        </motion.div>

        {/* Feature Contribution Bars */}
        <motion.div variants={itemVars} className="enterprise-card" style={{ overflowX: 'auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
            <h3 className="section-title" style={{ margin: 0 }}><TrendingUp size={18} color="var(--accent-primary)" /> Global Influence Vectors</h3>
            <div style={{ fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: 12, color: 'var(--text-secondary)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><div style={{ width: 8, height: 8, background: 'var(--accent-danger)', borderRadius: 2 }} /> INCREASES CHURN</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}><div style={{ width: 8, height: 8, background: 'var(--accent-success)', borderRadius: 2 }} /> DECREASES CHURN</div>
            </div>
          </div>
          
          {loading ? (
            <div className="importance-bar-container">
              {[1, 2, 3, 4, 5].map(i => (
                <div key={i} style={{ height: 24, background: 'var(--bg-input)', borderRadius: 4, animation: 'pulse 1.5s infinite opacity' }} />
              ))}
            </div>
          ) : importances ? (
            <div className="importance-bar-container">
              {importances
                .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
                .slice(0, 10)
                .map((fi, i) => (
                  <div key={i} className="importance-item">
                    <div className="importance-label">
                      {fi.feature}
                    </div>
                    <div className="importance-bar-wrapper">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${(Math.abs(fi.shap_value) / maxImportance) * 100}%` }}
                        transition={{ duration: 1, delay: i * 0.1, ease: 'easeOut' }}
                        className="importance-bar"
                        style={{
                          background: fi.shap_value >= 0 ? 'var(--accent-danger)' : 'var(--accent-success)',
                        }}
                      />
                    </div>
                    <div className="importance-value" style={{ color: fi.shap_value >= 0 ? 'var(--accent-danger)' : 'var(--accent-success)' }}>
                      {fi.shap_value >= 0 ? '+' : ''}{fi.shap_value.toFixed(3)}
                    </div>
                  </div>
                ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 0', color: 'var(--text-muted)' }}>
              <Info size={32} style={{ opacity: 0.1, marginBottom: 12 }} />
              <p>No SHAP vectors detected in the current stream.</p>
            </div>
          )}
        </motion.div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)', gap: 24, marginTop: 32 }}>
        <div className="enterprise-card">
          <div style={{ display: 'flex', gap: 10, color: 'var(--text-primary)', marginBottom: 10 }}>
            <Maximize2 size={16} /> <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>SHAP VALUES</span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            SHAP (SHapley Additive exPlanations) values provide a game-theoretic approach to explain individual predictions by distributing the "payout" (prediction) among the features.
          </p>
        </div>
        <div className="enterprise-card">
          <div style={{ display: 'flex', gap: 10, color: 'var(--text-primary)', marginBottom: 10 }}>
            <ListFilter size={16} /> <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>GLOBAL VS LOCAL</span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Global importance shows the trend across the entire dataset, while Local importance (shown here) explains exactly why this specific customer was scored this way.
          </p>
        </div>
        <div className="enterprise-card">
          <div style={{ display: 'flex', gap: 10, color: 'var(--text-primary)', marginBottom: 10 }}>
            <ShieldAlert size={16} /> <span style={{ fontSize: '0.8rem', fontWeight: 700 }}>BIAS MONITORING</span>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Ensuring transparency in decision making allows us to audit for unwanted algorithmic biases and maintain ethical AI standards in customer retention.
          </p>
        </div>
      </div>
    </motion.div>
  );
}
