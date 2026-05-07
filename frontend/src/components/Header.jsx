import React from 'react';
import CoralLogo from './CoralLogo';

const Header = ({ S }) => {
  return (
    <header style={S.header}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '16px' }}>
        <CoralLogo /> 
        <div style={S.eyebrow}>
          <div style={S.dot} />CORAL REEF MONITORING SYSTEM
        </div>
      </div>
      <h1 style={S.title}>AquaVision</h1>
      <p style={S.sub}>
        Hybrid AI analysis combining MobileNetV2 Vision and Gemini-2.5 Ecological reasoning.
      </p>
    </header>
  );
};

export default Header;