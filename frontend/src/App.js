// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
// Components import
import Header from './components/Header';
import Home from './components/Home';
import GameDetails from './components/GameDetails';
import GamesList from './components/GamesList';
import MyGames from './components/MyGames';
import Footer from './components/Footer';

const toastStyles = {
  position: "bottom-right",
  autoClose: 3000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  theme: "light",
};

// API base URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Axios configuration
axios.defaults.baseURL = API_BASE_URL;

function App() {
  const [loading, setLoading] = useState(true);

  // Initialize
  useEffect(() => {
    // Check API health
    const checkApiHealth = async () => {
      try {
        await axios.get('/api/health');
        setLoading(false);
      } catch (error) {
        console.error('API health check failed:', error);
        toast.error('Unable to connect to the recommendation service. Some features may be limited.');
        setLoading(false);
      }
    };

    checkApiHealth();
  }, []);

  return (
    <Router>
      <div className="app min-h-screen flex flex-col bg-gray-50">
        <Header />

        <main className="container mx-auto px-4 py-8 flex-grow">
          {loading ? (
            <div className="flex justify-center items-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
            </div>
          ) : (
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/games" element={<GamesList />} />
              <Route path="/games/:gameId" element={<GameDetails />} />
              <Route path="/my-games/:category" element={<MyGames />} />
              <Route path="/my-liked-games" element={<MyGames />} />
              <Route path="/my-purchased-games" element={<MyGames />} />
              <Route path="/my-recommended-games" element={<MyGames />} />
            </Routes>
          )}
        </main>

        <Footer />
        <ToastContainer
          position={toastStyles.position}
          autoClose={toastStyles.autoClose}
          hideProgressBar={toastStyles.hideProgressBar}
          closeOnClick={toastStyles.closeOnClick}
          pauseOnHover={toastStyles.pauseOnHover}
          draggable={toastStyles.draggable}
          theme={toastStyles.theme}
        />
      </div>
    </Router>
  );
}

export default App;