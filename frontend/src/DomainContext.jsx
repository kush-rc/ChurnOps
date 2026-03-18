import { createContext, useContext, useState } from 'react';

const DomainContext = createContext();

export function DomainProvider({ children }) {
  const [domain, setDomain] = useState('telco');
  return (
    <DomainContext.Provider value={{ domain, setDomain }}>
      {children}
    </DomainContext.Provider>
  );
}

export function useDomain() {
  return useContext(DomainContext);
}
