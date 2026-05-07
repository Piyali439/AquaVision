import React from 'react';

const Footer = ({ S }) => {
  return (
    <footer style={{
      marginTop: '60px',
      paddingTop: '24px',
      borderTop: '1px solid rgba(255,255,255,0.05)',
      textAlign: 'center'
    }}>
      <p style={{ fontSize: '12px', color: '#5a7a9a', letterSpacing: '0.05em' }}>
        DESIGNED & DEVELOPED BY <span style={{ color: '#00b4dc' }}>PIYALI GHOSH</span>
      </p>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '12px' }}>
        <a href="https://github.com/Piyali439" target="_blank" rel="noreferrer" style={{ fontSize: '10px', color: '#3a5a7a', textDecoration: 'none' }}>GITHUB</a>
        <a href="#" style={{ fontSize: '10px', color: '#3a5a7a', textDecoration: 'none' }}>DOCUMENTATION</a>
      </div>
    </footer>
  );
};

export default Footer;