// frontend/src/components/GameDetails.js
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-toastify';
import GameCard from './GameCard';

const GameDetails = ({ userPreferences, updatePreference }) => {
  const { gameId } = useParams();
  const [game, setGame] = useState(null);
  const [similarGames, setSimilarGames] = useState([]);
  const [loading, setLoading] = useState(true);

  const isLiked = userPreferences?.liked_games?.some(g => g.id === parseInt(gameId));
  const isDisliked = userPreferences?.disliked_games?.some(g => g.id === parseInt(gameId));

  useEffect(() => {
    const fetchGameData = async () => {
      setLoading(true);

      try {
        const [gameResponse, similarGamesResponse] = await Promise.all([
          axios.get(`/games/${gameId}`),
          axios.get(`/similar-games/${gameId}?count=4`)
        ]);

        setGame(gameResponse.data.game);
        setSimilarGames(similarGamesResponse.data.similar_games || []);
      } catch (error) {
        console.error('Error fetching game data:', error);
        toast.error('Failed to load game data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchGameData();
  }, [gameId]);

  const handleLike = () => {
    updatePreference(isLiked ? 'unlike' : 'like', parseInt(gameId));
  };

  const handleDislike = () => {
    updatePreference(isDisliked ? 'undislike' : 'dislike', parseInt(gameId));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!game) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4" role="alert">
        <strong className="font-bold">Error:</strong>
        <span className="block sm:inline"> Game not found or an error occurred.</span>
      </div>
    );
  }

  return (
    <div className="game-details">
      <div className="mb-4">
        <Link to="/games" className="text-blue-600 hover:text-blue-800">
          &larr; Back to Games
        </Link>
      </div>

      {/* Game header */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-8">
        <div className="relative h-64 md:h-80 bg-gray-800">
          <img
            src={`https://cdn.akamai.steamstatic.com/steam/apps/${game.id}/header.jpg`}
            alt={game.title}
            className="w-full h-full object-cover"
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = "https://via.placeholder.com/1200x400?text=Game+Image";
            }}
          />
        </div>

        <div className="p-6">
          <div className="flex flex-wrap justify-between items-start">
            <div className="mb-4 md:mb-0">
              <h1 className="text-3xl font-bold mb-2">{game.title}</h1>

              {game.tags && game.tags.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                  {game.tags.map((tag, index) => (
                    <span key={index} className="bg-gray-200 text-gray-700 px-2 py-1 rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="flex space-x-2">
              <button
                onClick={handleLike}
                className={`flex items-center space-x-1 px-4 py-2 rounded-md ${
                  isLiked 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
                </svg>
                <span>{isLiked ? 'Liked' : 'Like'}</span>
              </button>

              <button
                onClick={handleDislike}
                className={`flex items-center space-x-1 px-4 py-2 rounded-md ${
                  isDisliked 
                    ? 'bg-red-600 text-white' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M18 9.5a1.5 1.5 0 11-3 0v-6a1.5 1.5 0 013 0v6zM14 9.667v-5.43a2 2 0 00-1.105-1.79l-.05-.025A4 4 0 0011.055 2H5.64a2 2 0 00-1.962 1.608l-1.2 6A2 2 0 004.44 12H8v4a2 2 0 002 2 1 1 0 001-1v-.667a4 4 0 01.8-2.4l1.4-1.866a4 4 0 00.8-2.4z" />
                </svg>
                <span>{isDisliked ? 'Disliked' : 'Dislike'}</span>
              </button>
            </div>
          </div>

          {/* Game metadata */}
          <div className="mt-6 text-gray-600">
            {game.release_date && (
              <p className="mb-2">
                <strong>Release Date:</strong> {new Date(game.release_date).toLocaleDateString()}
              </p>
            )}

            {game.price_final !== undefined && (
              <p className="mb-2">
                <strong>Price:</strong> ${game.price_final.toFixed(2)}
              </p>
            )}

            {game.positive_ratio !== undefined && (
              <p className="mb-2">
                <strong>Positive Ratio:</strong> {(game.positive_ratio * 100).toFixed(0)}%
              </p>
            )}

            {/* Platform compatibility */}
            <div className="flex space-x-2 mt-4">
              {game.win && (
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">Windows</span>
              )}
              {game.mac && (
                <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-sm">Mac</span>
              )}
              {game.linux && (
                <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm">Linux</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Similar games */}
      {similarGames.length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Similar Games</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {similarGames.map(similarGame => (
              <GameCard
                key={similarGame.id}
                game={similarGame}
                userPreferences={userPreferences}
                updatePreference={updatePreference}
                showScore={false}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GameDetails;

