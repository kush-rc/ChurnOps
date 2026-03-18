import { useState, useEffect } from 'react';
import { useDomain } from '../DomainContext';
import { predictChurn, DOMAIN_FIELDS } from '../api';
import { 
  ClipboardCheck, 
  ChevronRight, 
  ChevronLeft, 
  AlertTriangle, 
  CheckCircle2, 
  Loader2,
  BrainCircuit,
  Info,
  Sparkles
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Predict() {
  const { domain } = useDomain();
  const config = DOMAIN_FIELDS[domain];
  const [formData, setFormData] = useState(() => buildDefaults(config));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [step, setStep] = useState(0);

  function buildDefaults(cfg) {
    const data = {};
    cfg.fields.forEach((f) => {
      if (f.type === 'select') data[f.name] = f.options[0];
      else data[f.name] = f.default ?? 0;
    });
    return data;
  }

  useEffect(() => {
    setFormData(buildDefaults(DOMAIN_FIELDS[domain]));
    setResult(null);
    setStep(0);
  }, [domain]);

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    try {
      const res = await predictChurn(domain, formData);
      setResult(res);
    } catch (err) {
      alert("Intelligence engine encounter an error. Please verify input parameters.");
    } finally {
      setLoading(false);
    }
  };

  // Group fields into 2 logical steps for wizard
  const fields = config.fields;
  const mid = Math.ceil(fields.length / 2);
  const fieldGroups = [fields.slice(0, mid), fields.slice(mid)];

  const nextStep = () => setStep(s => s + 1);
  const prevStep = () => setStep(s => s - 1);

  const containerVars = {
    hidden: { opacity: 0, x: 20 },
    visible: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -20 }
  };

  return (
    <div className="predict-container">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <BrainCircuit size={24} color="var(--accent-blue)" />
          <h2 style={{ margin: 0 }}>Predictive Analysis</h2>
        </div>
        <p>Execute high-precision churn inference using the current domain champion model.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 380px' : '1fr', gap: 32, transition: 'all 0.5s ease' }}>
        {/* Form Wizard */}
        <div className="enterprise-card" style={{ position: 'relative' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 24, alignItems: 'center' }}>
            <h3 style={{ fontSize: '1.2rem', margin: 0, display: 'flex', alignItems: 'center', gap: 10 }}>
              <Sparkles size={18} color="var(--accent-primary)" />
              {config.title}
            </h3>
            <div style={{ display: 'flex', gap: 8 }}>
              {[0, 1].map(i => (
                <div 
                  key={i} 
                  style={{ 
                    width: 40, 
                    height: 4, 
                    borderRadius: 2, 
                    background: i === step ? 'var(--accent-primary)' : 'var(--bg-input)',
                    transition: 'all 0.3s ease'
                  }} 
                />
              ))}
            </div>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              variants={containerVars}
              initial="hidden"
              animate="visible"
              exit="exit"
              transition={{ duration: 0.3 }}
            >
              <div className="form-grid">
                {fieldGroups[step].map((f) => (
                  <div key={f.name} className="form-group">
                    <label className="form-label">{f.label}</label>
                    {f.type === 'select' ? (
                      <select
                        className="form-select"
                        onChange={(e) => setFormData({ ...formData, [f.name]: isNaN(e.target.value) ? e.target.value : Number(e.target.value) })}
                        value={formData[f.name] || ''}
                      >
                        {f.options.map((opt) => (
                          <option key={opt} value={opt}>{String(opt)}</option>
                        ))}
                      </select>
                    ) : f.type === 'slider' ? (
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{f.min}</span>
                          <span style={{ fontSize: '0.85rem', color: 'var(--accent-primary)', fontWeight: 600 }}>{formData[f.name]}</span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{f.max}</span>
                        </div>
                        <input
                          type="range"
                          className="form-slider"
                          min={f.min}
                          max={f.max}
                          step={f.step || 1}
                          value={formData[f.name]}
                          onChange={(e) => setFormData({ ...formData, [f.name]: Number(e.target.value) })}
                        />
                      </div>
                    ) : (
                      <input
                        type="number"
                        className="form-input"
                        placeholder={`Enter ${f.label.toLowerCase()}...`}
                        onChange={(e) => setFormData({ ...formData, [f.name]: parseFloat(e.target.value) || 0 })}
                        value={formData[f.name] || ''}
                      />
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          </AnimatePresence>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 32 }}>
            <button 
              className="btn" 
              onClick={prevStep}
              disabled={step === 0}
            >
              <ChevronLeft size={18} /> Previous
            </button>

            {step === 0 ? (
              <button className="btn btn-primary" onClick={nextStep} style={{ minWidth: '120px' }}>
                Continue <ChevronRight size={18} />
              </button>
            ) : (
              <button 
                className="btn btn-primary" 
                onClick={handlePredict} 
                disabled={loading}
                style={{ minWidth: '160px' }}
              >
                {loading ? <Loader2 size={18} className="spinner" /> : (
                  <>Run Inference <ClipboardCheck size={18} /></>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Results Display */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9, x: 20 }}
              animate={{ opacity: 1, scale: 1, x: 0 }}
              className={`prediction-result ${result.label === 'Churned' ? 'danger' : 'safe'}`}
              style={{ height: 'fit-content', position: 'sticky', top: 32 }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                {result.label === 'Churned' ? <AlertTriangle color="var(--accent-danger)" size={20} /> : <CheckCircle2 color="var(--accent-success)" size={20} />}
                <span className="prediction-status" style={{ letterSpacing: '0.05em', fontWeight: 700 }}>
                  {result.label === 'Churned' ? 'ANALYSIS: CRITICAL RISK' : 'ANALYSIS: CUSTOMER STABLE'}
                </span>
              </div>

              <div className="prediction-probability" style={{ fontSize: '3.5rem' }}>
                {(result.churn_probability * 100).toFixed(1)}%
              </div>
              <div className="prediction-detail" style={{ marginBottom: 20, fontSize: '0.85rem' }}>
                {result.label === 'Churned' 
                  ? "This profile shows strong alignment with historically lost customers. Immediate retention strategy recommended."
                  : "Behavioral patterns suggest high loyalty. Maintaining current service standards is optimal." 
                }
              </div>

              <div className="progress-bar" style={{ height: 10, marginBottom: 24 }}>
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${result.churn_probability * 100}%` }}
                  className="progress-fill" 
                  style={{ 
                    background: result.label === 'Churned' ? 'var(--accent-danger)' : 'var(--accent-success)',
                  }} 
                />
              </div>

              <div style={{ padding: 16, background: 'var(--bg-input)', borderRadius: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Info size={12} />
                    <span>Inference confidence:</span>
                  </div>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{(result.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
