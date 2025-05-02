// frontend/src/components/Home.js
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
  const [isSearching, setIsSearching] = useState(false);

  // Load random games
  useEffect(() => {
    fetchRandomGames();
  }, []);

  // Save user actions to local storage
  useEffect(() => {
    localStorage.setItem('userGameActions', JSON.stringify(userActions));
  }, [userActions]);

  // Add debounce for search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchTerm) {
        handleSearch();
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [searchTerm]);

  const fetchRandomGames = async () => {
    setLoading(true);
    try {
      // Fetch games from the games API
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

    setIsSearching(true);
    try {
      const response = await axios.get(`/api/games?search=${encodeURIComponent(searchTerm)}`);
      setSearchResults(response.data.games || []);
    } catch (error) {
      console.error('Error searching games:', error);
      toast.error('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
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
  const hasUserActions = userActions.liked.length > 0 ||
                        userActions.purchased.length > 0 ||
                        userActions.recommended.length > 0;

  const heroSection = {
    background: 'linear-gradient(to right, #1f2937, #374151)',
    borderRadius: '0.5rem',
    padding: '2rem',
    marginBottom: '2rem',
    color: 'white', // 保持文字为白色
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
  };

  const primaryButtonStyle = {
    backgroundColor: '#2563eb',
    color: 'white',
    padding: '0.75rem 1.5rem',
    borderRadius: '0.375rem',
    border: 'none',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
  };

  const secondaryButtonStyle = {
    backgroundColor: '#4b5563',
    color: 'white',
    padding: '0.75rem 1.5rem',
    borderRadius: '0.375rem',
    border: 'none',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
  };

  const userActionsPanel = {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    padding: '1.5rem',
    marginBottom: '2rem',
    color: '#1f2937' // 深灰色文本
  };

  return (
    <div className="home">
      {/* Hero section - 保持白色文字在深色背景上 */}
      <div style={heroSection}>
          <h1 className="text-3xl md:text-4xl font-bold mb-4">Welcome to Steam Game Recommender</h1>
          <p className="text-lg opacity-90 mb-6">Discover new games based on your preferences and playing history.</p>

          {/* 搜索栏 */}
          <div className="max-w-2xl">
            <div className="flex flex-col md:flex-row gap-2">
              <input
                type="text"
                placeholder="Search games..."
                className="flex-1 px-4 py-3 rounded-md text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 w-full"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <button
                onClick={handleSearch}
                style={primaryButtonStyle}
                className="md:w-auto w-full"
              >
                Search
              </button>
            </div>
          </div>
        </div>

    {/* 用户操作面板 - 深色文字在浅色背景上 */}
    <div style={userActionsPanel}>
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex flex-wrap gap-3">
          <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full flex items-center">
            <span className="mr-1">❤️</span> {userActions.liked.length} Liked
          </span>
          <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full flex items-center">
            <span className="mr-1">🛒</span> {userActions.purchased.length} Purchased
          </span>
          <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full flex items-center">
            <span className="mr-1">👍</span> {userActions.recommended.length} Recommended
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-2">
          <button
            onClick={getRecommendations}
            style={{
              ...primaryButtonStyle,
              opacity: hasUserActions ? 1 : 0.5,
              cursor: hasUserActions ? 'pointer' : 'not-allowed'
            }}
            disabled={loading || !hasUserActions}
          >
            {loading ? 'Loading...' : 'Get Recommendations'}
          </button>

          <button
            onClick={resetActions}
            style={secondaryButtonStyle}
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
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: '1.5rem',
            marginBottom: '2rem'
          }}>
            {recommendations.map(game => (
              <div key={game.id} style={{ height: '100%' }}>
                <GameCard
                  key={game.id}
                  game={game}
                  onAction={handleGameAction}
                  userActions={userActions}
                  showScore={true}
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Games display area */}
      <section>
        <h2 className="text-2xl font-bold mb-4">
          {searchTerm ? 'Search Results' : 'Discover Games'}
        </h2>

        {/* Loading state */}
        {(loading || (isSearching && searchTerm)) && !displayGames.length ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
          </div>
        ) : displayGames.length === 0 ? (
          <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded mb-4">
            <p>No games found. Try a different search term or refresh the page.</p>
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: '1.5rem',
            marginBottom: '2rem'
          }}>
            {displayGames.map(game => (
              <div key={game.app_id} style={{ height: '100%' }}>
                <GameCard
                  game={game}
                  onAction={handleGameAction}
                  userActions={userActions}
                />
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default Home;