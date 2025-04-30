// frontend/src/components/Header.js
import React from 'react';
import { Link } from 'react-router-dom';

const Header = ({ user, onLogout }) => {
  return (
    <header className="bg-gray-800 text-white">
      <div className="container mx-auto px-4 py-4 flex flex-wrap items-center justify-between">
        <div className="flex items-center">
          <Link to="/" className="font-bold text-xl">Steam Game Recommender</Link>
        </div>

        <nav className="flex items-center">
          <ul className="flex space-x-6">
            <li>
              <Link to="/" className="hover:text-gray-300">Home</Link>
            </li>
            <li>
              <Link to="/games" className="hover:text-gray-300">Games</Link>
            </li>
            {user ? (
              <>
                <li>
                  <Link to="/profile" className="hover:text-gray-300">Profile</Link>
                </li>
                <li>
                  <button
                    onClick={onLogout}
                    className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded"
                  >
                    Logout
                  </button>
                </li>
              </>
            ) : (
              <>
                <li>
                  <Link to="/login" className="hover:text-gray-300">Login</Link>
                </li>
                <li>
                  <Link to="/register" className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded">
                    Register
                  </Link>
                </li>
              </>
            )}
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
