// frontend/src/components/Footer.js
import React from 'react';

const Footer = () => {
  const footerStyle = {
    backgroundColor: '#1f2937',
    color: 'white',
    padding: '2rem 0',
    marginTop: '3rem',
  };

  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 1rem',
  };

  const footerContentStyle = {
    display: 'flex',
    flexDirection: 'column',
    gap: '2rem',
  };

  const sectionStyle = {
    marginBottom: '1rem',
  };

  const headingStyle = {
    fontSize: '1.25rem',
    fontWeight: 'bold',
    marginBottom: '0.75rem',
  };

  const textStyle = {
    color: '#9ca3af',
  };

  const copyrightStyle = {
    borderTop: '1px solid #374151',
    marginTop: '2rem',
    paddingTop: '1.5rem',
    textAlign: 'center',
    color: '#9ca3af',
  };

  return (
    <footer style={footerStyle}>
      <div style={containerStyle}>
        <div style={footerContentStyle}>
          <div style={sectionStyle}>
            <h3 style={headingStyle}>Steam Game Recommender</h3>
            <p style={textStyle}>Personalized game recommendations based on your preferences.</p>
          </div>

          <div style={sectionStyle}>
            <h3 style={headingStyle}>About</h3>
            <p style={textStyle}>
              Powered by advanced machine learning algorithms<br />
              Includes KNN, SVD, and content-based recommendations
            </p>
          </div>
        </div>

        <div style={copyrightStyle}>
          &copy; {new Date().getFullYear()} Steam Game Recommender. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;