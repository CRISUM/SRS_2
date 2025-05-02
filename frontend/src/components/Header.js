// frontend/src/components/Header.js
import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [userActions, setUserActions] = useState({
    liked: [],
    purchased: [],
    recommended: []
  });

  const location = useLocation();

  // Load user actions from localStorage
  useEffect(() => {
    const loadUserActions = () => {
      const savedActions = localStorage.getItem('userGameActions');
      if (savedActions) {
        setUserActions(JSON.parse(savedActions));
      }
    };

    // Load on mount
    loadUserActions();

    // Set up event listener to catch changes
    window.addEventListener('storage', loadUserActions);

    // Clean up
    return () => {
      window.removeEventListener('storage', loadUserActions);
    };
  }, []);

  const headerStyle = {
    backgroundColor: '#1f2937',
    color: 'white',
    padding: '1rem 0',
  };

  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 1rem',
    position: 'relative'
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
    display: 'flex',
    alignItems: 'center'
  };

  const menuStyle = {
    display: 'flex',
    gap: '1.5rem',
    alignItems: 'center'
  };

  const linkStyle = {
    color: 'white',
    textDecoration: 'none',
    position: 'relative',
    padding: '0.25rem 0',
  };

  const activeLinkStyle = {
    ...linkStyle,
    borderBottom: '2px solid #60a5fa'
  };

  const dropdownButtonStyle = {
    ...linkStyle,
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem',
    cursor: 'pointer'
  };

  const dropdownMenuStyle = {
    position: 'absolute',
    top: '100%',
    right: 0,
    backgroundColor: 'white',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    borderRadius: '0.375rem',
    padding: '0.5rem 0',
    minWidth: '12rem',
    zIndex: 10,
    display: menuOpen ? 'block' : 'none'
  };

  const dropdownItemStyle = {
    display: 'flex',
    alignItems: 'center',
    padding: '0.5rem 1rem',
    fontSize: '0.875rem',
    color: '#1f2937',
    textDecoration: 'none',
    transition: 'background-color 0.2s',
    gap: '0.5rem'
  };

  const getActiveLinkStyle = (path) => {
    return location.pathname === path ? activeLinkStyle : linkStyle;
  };

  return (
    <header style={headerStyle}>
      <div style={containerStyle}>
        <nav style={navStyle}>
          <Link to="/" style={logoStyle}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
            </svg>
            Steam Game Recommender
          </Link>

          <div style={menuStyle}>
            <Link to="/" style={getActiveLinkStyle('/')}>
              Home
            </Link>
            <Link to="/games" style={getActiveLinkStyle('/games')}>
              Browse Games
            </Link>

            {/* My Games Dropdown */}
            <div style={{ position: 'relative' }}>
              <button
                style={dropdownButtonStyle}
                onClick={() => setMenuOpen(!menuOpen)}
                onBlur={() => setTimeout(() => setMenuOpen(false), 100)}
              >
                My Games
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>

              <div style={dropdownMenuStyle}>
                <Link
                  to="/my-liked-games"
                  style={dropdownItemStyle}
                  onClick={() => setMenuOpen(false)}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
                  </svg>
                  Liked Games
                  <span className="bg-gray-100 text-gray-700 text-xs px-2 py-0.5 rounded-full ml-auto">
                    {userActions.liked.length}
                  </span>
                </Link>
                <Link
                  to="/my-purchased-games"
                  style={dropdownItemStyle}
                  onClick={() => setMenuOpen(false)}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-green-500" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3zM16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
                  </svg>
                  My Library
                  <span className="bg-gray-100 text-gray-700 text-xs px-2 py-0.5 rounded-full ml-auto">
                    {userActions.purchased.length}
                  </span>
                </Link>
                <Link
                  to="/my-recommended-games"
                  style={dropdownItemStyle}
                  onClick={() => setMenuOpen(false)}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
                  </svg>
                  Recommended
                  <span className="bg-gray-100 text-gray-700 text-xs px-2 py-0.5 rounded-full ml-auto">
                    {userActions.recommended.length}
                  </span>
                </Link>
              </div>
            </div>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;