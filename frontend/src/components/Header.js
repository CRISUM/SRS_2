// frontend/src/components/Header.js
import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  const headerStyle = {
    backgroundColor: '#1f2937',
    color: 'white',
    padding: '1rem 0',
  };

  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 1rem',
  };

  const navStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  };

  const logoStyle = {
    fontWeight: 'bold',
    fontSize: '1.25rem',
    color: 'white',
    textDecoration: 'none',
  };

  const menuStyle = {
    display: 'flex',
    gap: '1.5rem',
  };

  const linkStyle = {
    color: 'white',
    textDecoration: 'none',
  };

  return (
    <header style={headerStyle}>
      <div style={containerStyle}>
        <nav style={navStyle}>
          <Link to="/" style={logoStyle}>
            Steam Game Recommender
          </Link>

          <div style={menuStyle}>
            <Link to="/" style={linkStyle}>
              Home
            </Link>
            <Link to="/games" style={linkStyle}>
              Browse Games
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;