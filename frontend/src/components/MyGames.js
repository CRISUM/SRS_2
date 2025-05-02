// frontend/src/components/MyGames.js
import React, { useState, useEffect } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-toastify';
import GameCard from './GameCard';

const MyGames = () => {
  const { category } = useParams();
  const navigate = useNavigate();
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filteredGames, setFilteredGames] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  // Selected tab
  const [selectedCategory, setSelectedCategory] = useState(category || 'liked');

  // User actions state - Load from localStorage
  const [userActions, setUserActions] = useState(() => {
    const savedActions = localStorage.getItem('userGameActions');
    return savedActions ? JSON.parse(savedActions) : {
      liked: [],
      purchased: [],
      recommended: []
    };
  });

  // Load all games and filter based on user actions
  useEffect(() => {
    fetchAllGames();
  }, []);

  // Update filtered games when category or search changes
  useEffect(() => {
    filterGames();
  }, [selectedCategory, searchTerm, games, userActions]);

  // Update URL when category changes
  useEffect(() => {
    navigate(`/my-games/${selectedCategory}`);
  }, [selectedCategory, navigate]);

  const fetchAllGames = async () => {
    setLoading(true);
    try {
      // Get all games the user has interacted with
      const gameIds = [
        ...userActions.liked,
        ...userActions.purchased,
        ...userActions.recommended
      ];

      // Remove duplicates
      const uniqueGameIds = [...new Set(gameIds)];

      if (uniqueGameIds.length === 0) {
        setGames([]);
        setLoading(false);
        return;
      }

      // Fetch details for all these games
      const gameDetailsPromises = uniqueGameIds.map(id =>
        axios.get(`/api/games/${id}`)
          .then(response => response.data.game)
          .catch(error => {
            console.error(`Failed to fetch game ${id}:`, error);
            return null;
          })
      );

      const gameDetails = await Promise.all(gameDetailsPromises);
      setGames(gameDetails.filter(game => game !== null));
    } catch (error) {
      console.error('Error fetching games:', error);
      toast.error('Failed to load your games. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const filterGames = () => {
    // Get IDs from selected category
    let categoryIds = [];
    switch(selectedCategory) {
      case 'liked':
        categoryIds = userActions.liked;
        break;
      case 'purchased':
        categoryIds = userActions.purchased;
        break;
      case 'recommended':
        categoryIds = userActions.recommended;
        break;
      default:
        categoryIds = [];
    }

    // Filter games by ID and search term
    let filtered = games.filter(game => categoryIds.includes(game.id));

    if (searchTerm) {
      filtered = filtered.filter(game =>
        game.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (game.tags && game.tags.some(tag =>
          tag.toLowerCase().includes(searchTerm.toLowerCase())
        ))
      );
    }

    setFilteredGames(filtered);
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

      // Save to localStorage
      localStorage.setItem('userGameActions', JSON.stringify(newActions));
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

  // Get category information for UI
  const getCategoryInfo = () => {
    switch(selectedCategory) {
      case 'liked':
        return {
          title: 'Liked Games',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
            </svg>
          ),
          color: 'text-red-600',
          emptyMessage: 'You haven\'t liked any games yet.'
        };
      case 'purchased':
        return {
          title: 'My Library',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
              <path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3zM16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
            </svg>
          ),
          color: 'text-green-600',
          emptyMessage: 'You haven\'t added any games to your library yet.'
        };
      case 'recommended':
        return {
          title: 'Recommended Games',
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
              <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
            </svg>
          ),
          color: 'text-blue-600',
          emptyMessage: 'You haven\'t recommended any games yet.'
        };
      default:
        return {
          title: 'My Games',
          icon: null,
          color: 'text-gray-700',
          emptyMessage: 'You haven\'t interacted with any games yet.'
        };
    }
  };

  const categoryInfo = getCategoryInfo();

  return (
    <div className="my-games">
      <div className="mb-6">
        <div className="flex items-center mb-2">
          {categoryInfo.icon && <span className={categoryInfo.color}>{categoryInfo.icon}</span>}
          <h1 className={`text-3xl font-bold ${categoryInfo.color}`}>{categoryInfo.title}</h1>
        </div>

        <Link to="/games" className="text-blue-600 hover:text-blue-800 inline-flex items-center text-xs">

          Back to Browse Games
        </Link>
      </div>


      {/* Category tabs */}
      <div className="flex border-b border-gray-200 mb-6 overflow-x-auto">
        <button
          onClick={() => setSelectedCategory('liked')}
          className={`px-3 py-1 text-xs font-medium flex items-center ${
            selectedCategory === 'liked'
              ? 'border-b-2 border-red-500 text-red-600'
              : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
          </svg>
          Liked ({userActions.liked.length})
        </button>

        <button
          onClick={() => setSelectedCategory('purchased')}
          className={`px-3 py-1 text-xs font-medium flex items-center ${
            selectedCategory === 'purchased'
              ? 'border-b-2 border-green-500 text-green-600'
              : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" viewBox="0 0 20 20" fill="currentColor">
            <path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3zM16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
          </svg>
          Library ({userActions.purchased.length})
        </button>

        <button
          onClick={() => setSelectedCategory('recommended')}
          className={`px-3 py-1 text-xs font-medium flex items-center ${
            selectedCategory === 'recommended'
              ? 'border-b-2 border-blue-500 text-blue-600'
              : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" viewBox="0 0 20 20" fill="currentColor">
            <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
          </svg>
          Recommended ({userActions.recommended.length})
        </button>
      </div>

      {/* Search */}
      <div className="mb-6">
        <div className="relative">
          <input
            type="text"
            placeholder={`Search in ${categoryInfo.title.toLowerCase()}...`}
            className="w-full px-4 py-2 pr-10 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Loading state */}
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
        </div>
      ) : filteredGames.length === 0 ? (
        <div className="bg-gray-100 rounded-lg p-6 text-center">
          <p className="text-gray-600 mb-4">{categoryInfo.emptyMessage}</p>
          <Link to="/games" className="inline-block bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md">
            Browse Games
          </Link>
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2rem'
        }}>
          {filteredGames.map(game => (
            <GameCard
              key={game.id}
              game={game}
              onAction={handleGameAction}
              userActions={userActions}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default MyGames;