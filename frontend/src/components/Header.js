// frontend/src/components/Header.js
import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
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
              <Link to="/games" className="hover:text-gray-300">Browse Games</Link>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;