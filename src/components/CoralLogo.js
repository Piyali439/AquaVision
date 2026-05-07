const CoralLogo = () => (
  <svg width="40" height="40" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M50 5L89 27.5V72.5L50 95L11 72.5V27.5L50 5Z" stroke="#00b4dc" strokeWidth="2" fill="rgba(0,180,220,0.05)"/>
    <path d="M50 75V45M50 60L65 50M50 55L35 40" stroke="#00e5a0" strokeWidth="5" strokeLinecap="round"/>
    <circle cx="50" cy="35" r="6" fill="#00b4dc">
      <animate attributeName="opacity" values="1;0.4;1" dur="2s" repeatCount="indefinite" />
    </circle>
  </svg>
);

export default CoralLogo;