// frontend/src/components/Home.js - Fixed API calls
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-toastify';
import GameCard from './GameCard';

const Home = () => {
  const [randomGames, setRandomGames] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [userActions, setUserActions] = useState(() => {
    // Load user actions from local storage
    const savedActions = localStorage.getItem('userGameActions');
    return savedActions ? JSON.parse(savedActions) : {
      liked: [],
      purchased: [],
      recommended: []
    };
  });
  const [showRecommendations, setShowRecommendations] = useState(false);

  // Load random games
  useEffect(() => {
    fetchRandomGames();
  }, []);

  // Save user actions to local storage
  useEffect(() => {
    localStorage.setItem('userGameActions', JSON.stringify(userActions));
  }, [userActions]);

  const fetchRandomGames = async () => {
    setLoading(true);
    try {
      // Fetch games from the games API - FIXED path
      const response = await axios.get('/api/games?limit=20');
      setRandomGames(response.data.games || []);
    } catch (error) {
      console.error('Error fetching random games:', error);
      toast.error('Failed to load games. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      const response = await axios.get(`/api/games?search=${encodeURIComponent(searchTerm)}`);
      setSearchResults(response.data.games || []);
    } catch (error) {
      console.error('Error searching games:', error);
      toast.error('Search failed. Please try again.');
    }
  };

  const handleGameAction = (gameId, actionType) => {
    setUserActions(prev => {
      // Create a copy of the action arrays
      const newActions = { ...prev };
      const gameIdNum = parseInt(gameId);

      // Update the appropriate array based on action type
      switch (actionType) {
        case 'like':
          if (!newActions.liked.includes(gameIdNum)) {
            newActions.liked = [...newActions.liked, gameIdNum];
          }
          break;
        case 'buy':
          if (!newActions.purchased.includes(gameIdNum)) {
            newActions.purchased = [...newActions.purchased, gameIdNum];
          }
          break;
        case 'recommend':
          if (!newActions.recommended.includes(gameIdNum)) {
            newActions.recommended = [...newActions.recommended, gameIdNum];
          }
          break;
        case 'unlike':
          newActions.liked = newActions.liked.filter(id => id !== gameIdNum);
          break;
        case 'unbuy':
          newActions.purchased = newActions.purchased.filter(id => id !== gameIdNum);
          break;
        case 'unrecommend':
          newActions.recommended = newActions.recommended.filter(id => id !== gameIdNum);
          break;
        default:
          break;
      }

      return newActions;
    });

    // Show notification of the action
    const actionMessages = {
      like: 'Game added to liked games',
      buy: 'Game added to your library',
      recommend: 'Game added to recommended games',
      unlike: 'Game removed from liked games',
      unbuy: 'Game removed from your library',
      unrecommend: 'Game removed from recommended games'
    };

    toast.success(actionMessages[actionType] || 'Action recorded');
  };

  const getRecommendations = async () => {
    setLoading(true);
    try {
      // Send user action history to backend for recommendations
      const response = await axios.post('/api/custom-recommendations', userActions);
      setRecommendations(response.data.recommendations || []);
      setShowRecommendations(true);

      // Show success message
      toast.success('Recommendations generated based on your preferences!');
    } catch (error) {
      console.error('Error getting recommendations:', error);
      toast.error('Failed to generate recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetActions = () => {
    // Clear all user actions
    setUserActions({
      liked: [],
      purchased: [],
      recommended: []
    });
    setShowRecommendations(false);
    toast.info('Your preferences have been reset');
  };

  // Determine which games to display
  const displayGames = searchTerm ? searchResults : randomGames;

  return (
    <div className="home">
      {/* Search bar */}
      <div className="mb-6 flex gap-2">
        <input
          type="text"
          placeholder="Search games..."
          className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        />
        <button
          onClick={handleSearch}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md"
        >
          Search
        </button>
      </div>

      {/* User action counts and action buttons */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <div className="flex flex-wrap items-center justify-between">
          <div className="flex space-x-4 mb-2 md:mb-0">
            <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full">
              ❤️ {userActions.liked.length} Liked
            </span>
            <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full">
              🛒 {userActions.purchased.length} Purchased
            </span>
            <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
              👍 {userActions.recommended.length} Recommended
            </span>
          </div>

          <div className="flex space-x-2">
            <button
              onClick={getRecommendations}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-md"
              disabled={loading || (userActions.liked.length === 0 && userActions.purchased.length === 0 && userActions.recommended.length === 0)}
            >
              {loading ? 'Loading...' : 'Get Recommendations'}
            </button>

            <button
              onClick={resetActions}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-md"
            >
              Reset Preferences
            </button>
          </div>
        </div>
      </div>

      {/* Recommendations area */}
      {showRecommendations && recommendations.length > 0 && (
        <section className="mb-10">
          <h2 className="text-2xl font-bold mb-4">Your Personalized Recommendations</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {recommendations.map(game => (
              <GameCard
                key={game.id}
                game={game}
                onAction={handleGameAction}
                userActions={userActions}
                showScore={true}
              />
            ))}
          </div>
        </section>
      )}

      {/* Games display area */}
      <section>
        <h2 className="text-2xl font-bold mb-4">
          {searchTerm ? 'Search Results' : 'Discover Games'}
        </h2>

        {loading && displayGames.length === 0 ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
          </div>
        ) : displayGames.length === 0 ? (
          <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded mb-4">
            <p>No games found. Try a different search term or refresh the page.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {displayGames.map(game => (
              <GameCard
                key={game.id}
                game={game}
                onAction={handleGameAction}
                userActions={userActions}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default Home;