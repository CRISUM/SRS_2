// src/App.js - Updated API Configuration
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
import Footer from './components/Footer';

// API base URL - FIXED to remove the duplicate /api
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Axios configuration
axios.defaults.baseURL = API_BASE_URL;

function App() {
  const [popularGames, setPopularGames] = useState([]);
  const [loading, setLoading] = useState(true);

  // Initialize - load popular games
  useEffect(() => {
    fetchPopularGames();
  }, []);

  // Fetch popular games
  const fetchPopularGames = async () => {
    try {
      const response = await axios.get('/api/popular-games?count=10');
      setPopularGames(response.data.popular_games || []);
    } catch (error) {
      console.error('Error fetching popular games:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Router>
      <div className="app min-h-screen flex flex-col">
        <Header />

        <main className="container mx-auto px-4 py-6 flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/games" element={<GamesList />} />
            <Route path="/games/:gameId" element={<GameDetails />} />
          </Routes>
        </main>
        
        <Footer />
        <ToastContainer position="bottom-right" autoClose={3000} />
      </div>
    </Router>
  );
}

export default App;