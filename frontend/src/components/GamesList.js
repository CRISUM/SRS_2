// frontend/src/components/GamesList.js - Fixed API calls
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import GameCard from './GameCard';

const GamesList = () => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [limit, setLimit] = useState(12);

  // Search state
  const [searchTerm, setSearchTerm] = useState('');
  const [searchTimeout, setSearchTimeout] = useState(null);

  // User actions state - Load from localStorage
  const [userActions, setUserActions] = useState(() => {
    const savedActions = localStorage.getItem('userGameActions');
    return savedActions ? JSON.parse(savedActions) : {
      liked: [],
      purchased: [],
      recommended: []
    };
  });

  // Load games on initial render and when page or limit changes
  useEffect(() => {
    fetchGames(currentPage, limit, searchTerm);
  }, [currentPage, limit]);

  // Debounce search input
  useEffect(() => {
    if (searchTimeout) {
      clearTimeout(searchTimeout);
    }

    const timeout = setTimeout(() => {
      fetchGames(1, limit, searchTerm);
      setCurrentPage(1);
    }, 500);

    setSearchTimeout(timeout);

    return () => clearTimeout(timeout);
  }, [searchTerm, limit]);

  // Save user actions to localStorage when they change
  useEffect(() => {
    localStorage.setItem('userGameActions', JSON.stringify(userActions));
  }, [userActions]);

  const fetchGames = async (page, pageLimit, search) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get('/api/games', {
        params: { page, limit: pageLimit, search }
      });

      setGames(response.data.games || []);
      setTotalPages(response.data.pagination.total_pages || 1);
    } catch (err) {
      console.error('Error fetching games:', err);
      setError('Failed to load games. Please try again later.');
      toast.error('Failed to load games. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
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

  return (
    <div className="games-list">
      <h1 className="text-3xl font-bold mb-6">Game Library</h1>

      <div className="mb-6">
        <input
          type="text"
          placeholder="Search games..."
          className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={searchTerm}
          onChange={handleSearchChange}
        />
      </div>

      {loading && games.length === 0 ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4" role="alert">
          <span className="block sm:inline">{error}</span>
        </div>
      ) : games.length === 0 ? (
        <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded mb-4" role="alert">
          <span className="block sm:inline">No games found. Try a different search term.</span>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {games.map(game => (
              <GameCard
                key={game.id}
                game={game}
                onAction={handleGameAction}
                userActions={userActions}
              />
            ))}
          </div>

          {/* Pagination */}
          <div className="mt-8 flex justify-center">
            <nav className="inline-flex rounded-md shadow">
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className={`px-4 py-2 text-sm font-medium rounded-l-md ${
                  currentPage === 1
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                Previous
              </button>

              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                // Calculate page numbers to show
                let pageNum;
                if (totalPages <= 5) {
                  pageNum = i + 1;
                } else if (currentPage <= 3) {
                  pageNum = i + 1;
                } else if (currentPage >= totalPages - 2) {
                  pageNum = totalPages - 4 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }

                return (
                  <button
                    key={i}
                    onClick={() => handlePageChange(pageNum)}
                    className={`px-4 py-2 text-sm font-medium ${
                      currentPage === pageNum
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {pageNum}
                  </button>
                );
              })}

              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className={`px-4 py-2 text-sm font-medium rounded-r-md ${
                  currentPage === totalPages
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                Next
              </button>
            </nav>
          </div>
        </>
      )}
    </div>
  );
};

export default GamesList;