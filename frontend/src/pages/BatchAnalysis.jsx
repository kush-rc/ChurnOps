import { useState, useRef, useMemo } from 'react';
import { useDomain } from '../DomainContext';
import { uploadBatchCSV } from '../api';
import { motion } from 'framer-motion';
import {
  Upload,
  FileSpreadsheet,
  AlertTriangle,
  CheckCircle2,
  Users,
  TrendingUp,
  ShieldAlert,
  Download,
  ChevronLeft,
  ChevronRight,
  X,
  BarChart3,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';

const ROWS_PER_PAGE = 15;

export default function BatchAnalysis() {
  const { domain } = useDomain();
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [page, setPage] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');

  const handleFile = (f) => {
    if (f && f.name.endsWith('.csv')) {
      setFile(f);
      setError(null);
    } else {
      setError('Please upload a .csv file');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await uploadBatchCSV(domain, file);
      setResults(data);
      setPage(0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResults(null);
    setError(null);
    setPage(0);
    setSearchTerm('');
  };

  // Filtered + paginated predictions
  const filteredPredictions = useMemo(() => {
    if (!results?.predictions) return [];
    if (!searchTerm) return results.predictions;
    return results.predictions.filter((p, i) =>
      JSON.stringify(p).toLowerCase().includes(searchTerm.toLowerCase()) ||
      String(i + 1).includes(searchTerm)
    );
  }, [results, searchTerm]);

  const totalPages = Math.ceil(filteredPredictions.length / ROWS_PER_PAGE);
  const paginatedRows = filteredPredictions.slice(page * ROWS_PER_PAGE, (page + 1) * ROWS_PER_PAGE);

  const exportCSV = () => {
    if (!results?.predictions) return;
    const header = 'Row,Prediction,Probability,Label,Confidence\n';
    const rows = results.predictions.map((p, i) =>
      `${i + 1},${p.prediction},${p.churn_probability.toFixed(4)},${p.label},${p.confidence.toFixed(4)}`
    ).join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `churnops_batch_results_${domain}.csv`;
    a.click();
  };

  const PIE_COLORS = ['var(--accent-success)', 'var(--accent-warning)', 'var(--accent-danger)'];
  const resolveColor = (cssVar) => {
    const style = getComputedStyle(document.documentElement);
    return style.getPropertyValue(cssVar.replace('var(', '').replace(')', '')).trim();
  };

  // ─── Upload State ─────────────────────────────────────────────────
  if (!results && !loading) {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="page-header">
          <h2>Batch Analysis</h2>
          <p>Upload a CSV file of customer data to run churn predictions on your entire dataset.</p>
        </div>

        <div className="enterprise-card" style={{ maxWidth: 640, margin: '0 auto' }}>
          <div
            className={`upload-zone${dragOver ? ' dragover' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
            <div className="upload-zone-icon">
              <Upload size={40} />
            </div>
            <div className="upload-zone-text">
              Drag and drop your CSV file here, or click to browse
            </div>
            <div className="upload-zone-hint">
              Accepted format: .csv — Each row represents one customer
            </div>

            {file && (
              <div className="upload-zone-file">
                <FileSpreadsheet size={16} />
                {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </div>
            )}
          </div>

          {error && (
            <div className="enterprise-card" style={{ marginTop: 24, padding: 16, border: '1px solid rgba(239, 68, 68, 0.2)', background: 'rgba(239, 68, 68, 0.05)' }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <AlertTriangle size={20} color="var(--accent-danger)" style={{ marginTop: 2 }} />
                <div>
                  <div style={{ color: 'var(--accent-danger)', fontWeight: 600, marginBottom: 4 }}>Validation Failed</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    {error}
                  </div>
                  <div style={{ marginTop: 12, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Tip: Ensure your CSV headers match the required fields for the <strong>{domain.toUpperCase()}</strong> domain. You can use the reference file in <code>data/reference/</code> as a guide.
                  </div>
                </div>
              </div>
            </div>
          )}

          <button
            className="btn btn-primary btn-full"
            style={{ marginTop: 20 }}
            disabled={!file}
            onClick={handleUpload}
          >
            <BarChart3 size={16} />
            Run Batch Predictions
          </button>
        </div>
      </motion.div>
    );
  }

  // ─── Processing State ─────────────────────────────────────────────
  if (loading) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="page-header">
          <h2>Batch Analysis</h2>
        </div>
        <div className="enterprise-card">
          <div className="processing-overlay">
            <div className="processing-spinner" />
            <h3 style={{ marginBottom: 8 }}>Processing {file?.name}</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Running churn predictions on all rows...
            </p>
          </div>
        </div>
      </motion.div>
    );
  }

  // ─── Results State ────────────────────────────────────────────────
  const { total, churned_count, churn_rate, avg_probability, probability_distribution, confidence_breakdown, top_risk_customers } = results;

  const pieData = confidence_breakdown ? [
    { name: 'Low Risk', value: confidence_breakdown.low || 0 },
    { name: 'Medium Risk', value: confidence_breakdown.medium || 0 },
    { name: 'High Risk', value: confidence_breakdown.high || 0 },
  ] : [];

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2>Batch Results</h2>
          <p>{total} customers analyzed from {file?.name}</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn" onClick={exportCSV}><Download size={14} /> Export CSV</button>
          <button className="btn" onClick={handleReset}><X size={14} /> New Upload</button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header"><Users size={18} color="var(--accent-primary)" /><span className="metric-label">Total Records</span></div>
          <div className="metric-value">{total}</div>
        </div>
        <div className="metric-card">
          <div className="metric-header"><AlertTriangle size={18} color="var(--accent-danger)" /><span className="metric-label">Churned</span></div>
          <div className="metric-value">{churned_count}</div>
          <div className="metric-sub">{(churn_rate * 100).toFixed(1)}% churn rate</div>
        </div>
        <div className="metric-card">
          <div className="metric-header"><TrendingUp size={18} color="var(--accent-warning)" /><span className="metric-label">Avg Probability</span></div>
          <div className="metric-value">{(avg_probability * 100).toFixed(1)}%</div>
        </div>
        <div className="metric-card">
          <div className="metric-header"><ShieldAlert size={18} color="var(--accent-danger)" /><span className="metric-label">High Risk</span></div>
          <div className="metric-value">{confidence_breakdown?.high || 0}</div>
          <div className="metric-sub">probability &gt; 70%</div>
        </div>
      </div>

      {/* Charts Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20, marginBottom: 24 }}>
        {/* Probability Distribution */}
        <div className="enterprise-card">
          <h3 className="section-title"><BarChart3 size={16} /> Probability Distribution</h3>
          <div className="chart-container">
            <ResponsiveContainer>
              <BarChart data={probability_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                <XAxis dataKey="bin" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
                <Tooltip
                  contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 8, fontSize: 13 }}
                  labelStyle={{ color: 'var(--text-primary)' }}
                />
                <Bar dataKey="count" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Risk Segmentation */}
        <div className="enterprise-card">
          <h3 className="section-title"><ShieldAlert size={16} /> Risk Segmentation</h3>
          <div className="chart-container">
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  innerRadius={50}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {pieData.map((_, idx) => (
                    <Cell key={idx} fill={resolveColor(PIE_COLORS[idx])} />
                  ))}
                </Pie>
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Top Risk Customers */}
      {top_risk_customers?.length > 0 && (
        <div className="enterprise-card" style={{ marginBottom: 24 }}>
          <h3 className="section-title"><AlertTriangle size={16} /> Top Risk Customers</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Row</th>
                <th>Churn Probability</th>
                <th>Risk Level</th>
                <th>Label</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {top_risk_customers.map((c, i) => (
                <tr key={i}>
                  <td>#{c.row_index}</td>
                  <td className="highlight">{(c.churn_probability * 100).toFixed(1)}%</td>
                  <td><span className={`risk-badge ${c.churn_probability > 0.7 ? 'high' : c.churn_probability > 0.4 ? 'medium' : 'low'}`}>
                    {c.churn_probability > 0.7 ? 'High' : c.churn_probability > 0.4 ? 'Medium' : 'Low'}
                  </span></td>
                  <td>{c.label}</td>
                  <td>{(c.confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Full Results Table */}
      <div className="enterprise-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 className="section-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>
            <CheckCircle2 size={16} /> All Predictions
          </h3>
          <input
            type="text"
            className="form-input"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); setPage(0); }}
            style={{ width: 200 }}
          />
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Row</th>
              <th>Prediction</th>
              <th>Churn Probability</th>
              <th>Label</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {paginatedRows.map((p, i) => (
              <tr key={i}>
                <td>{page * ROWS_PER_PAGE + i + 1}</td>
                <td>{p.prediction}</td>
                <td>{(p.churn_probability * 100).toFixed(2)}%</td>
                <td><span className={`risk-badge ${p.churn_probability > 0.7 ? 'high' : p.churn_probability > 0.4 ? 'medium' : 'low'}`}>
                  {p.label}
                </span></td>
                <td>{(p.confidence * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        {totalPages > 1 && (
          <div className="pagination">
            <button disabled={page === 0} onClick={() => setPage(p => p - 1)}><ChevronLeft size={14} /></button>
            {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
              const pg = totalPages <= 7 ? i : Math.max(0, Math.min(page - 3, totalPages - 7)) + i;
              return (
                <button key={pg} className={pg === page ? 'active' : ''} onClick={() => setPage(pg)}>
                  {pg + 1}
                </button>
              );
            })}
            <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}><ChevronRight size={14} /></button>
          </div>
        )}
      </div>
    </motion.div>
  );
}
