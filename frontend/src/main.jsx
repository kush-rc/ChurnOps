import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { DomainProvider } from './DomainContext';
import App from './App';
import './index.css';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <DomainProvider>
        <App />
      </DomainProvider>
    </BrowserRouter>
  </StrictMode>
);
