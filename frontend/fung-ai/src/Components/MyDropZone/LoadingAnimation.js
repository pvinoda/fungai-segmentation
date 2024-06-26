import React from 'react';
import './MyDropZone.css'

const LoadingAnimation = () => {
  return (
    <div className="loading-animation">
      <div className="loading-spinner"></div>
      <p>Loading...</p>
    </div>
  );
};

export default LoadingAnimation;