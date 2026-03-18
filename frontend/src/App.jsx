import { useState, useEffect } from 'react';
import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { useDomain } from './DomainContext';
import { DOMAINS } from './api';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import ModelComparison from './pages/ModelComparison';
import Explainability from './pages/Explainability';
import BatchAnalysis from './pages/BatchAnalysis';
import { 
  LayoutDashboard, 
  Target, 
  BarChart3, 
  Binary, 
  FileSpreadsheet,
  Settings, 
  ChevronRight,
  Activity,
  Sun,
  Moon,
  Zap
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const { domain, setDomain } = useDomain();
  const location = useLocation();

  // Dark mode state — persisted to localStorage
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('churnops-theme');
    return saved === 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('churnops-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const navItems = [
    { to: '/', icon: <LayoutDashboard size={18} />, label: 'Dashboard' },
    { to: '/predict', icon: <Target size={18} />, label: 'Predict' },
    { to: '/batch', icon: <FileSpreadsheet size={18} />, label: 'Batch Analysis' },
    { to: '/models', icon: <BarChart3 size={18} />, label: 'Models' },
    { to: '/explain', icon: <Binary size={18} />, label: 'Explainability' },
  ];

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <h1>
            ChurnOps
          </h1>
        </div>

        <nav className="nav-links">
          {navItems.map((item) => (
            <div className="nav-item" key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                {({ isActive }) => (
                  <>
                    {isActive && <div className="nav-active-pill" />}
                    <span style={{ position: 'relative', zIndex: 2, display: 'flex', alignItems: 'center', gap: 12 }}>
                      {item.icon}
                      {item.label}
                    </span>
                  </>
                )}
              </NavLink>
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="domain-selector">
            <label className="domain-label">
              <Settings size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />
              Intelligence Suite
            </label>
            <div className="domain-select-wrapper">
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
              >
                {Object.entries(DOMAINS).map(([key, d]) => (
                  <option key={key} value={key}>{d.label}</option>
                ))}
              </select>
              <ChevronRight className="domain-select-icon" size={14} />
            </div>
          </div>

          {/* Theme Toggle */}
          <div className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? <Moon size={14} /> : <Sun size={14} />}
            <div className="theme-toggle-track">
              <div className="theme-toggle-thumb" />
            </div>
            <span>{darkMode ? 'Dark' : 'Light'}</span>
          </div>

          <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.7rem', color: 'var(--accent-success)' }}>
            <Zap size={10} fill="currentColor" />
            <span>SYSTEM ACTIVE</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
          >
            <Routes location={location}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/batch" element={<BatchAnalysis />} />
              <Route path="/models" element={<ModelComparison />} />
              <Route path="/explain" element={<Explainability />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
