
// frontend/src/components/Footer.js
import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white py-6">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between">
          <div className="mb-4 md:mb-0">
            <h3 className="text-lg font-semibold mb-2">Steam Game Recommender</h3>
            <p className="text-gray-400">Personalized game recommendations based on your preferences</p>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-2">About</h3>
            <p className="text-gray-400">
              Powered by advanced machine learning algorithms<br />
              Includes KNN, SVD, and content-based recommendations
            </p>
          </div>
        </div>

        <div className="mt-8 border-t border-gray-700 pt-4 text-center text-gray-400">
          &copy; {new Date().getFullYear()} Steam Game Recommender. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
