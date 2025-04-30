// frontend/src/components/GameCard.js
import React from 'react';
import { Link } from 'react-router-dom';

const GameCard = ({ game, userPreferences, updatePreference, showScore, showPopularity }) => {
  const isLiked = userPreferences?.liked_games?.some(g => g.id === game.id);
  const isDisliked = userPreferences?.disliked_games?.some(g => g.id === game.id);

  const handleLike = () => {
    updatePreference(isLiked ? 'unlike' : 'like', game.id);
  };

  const handleDislike = () => {
    updatePreference(isDisliked ? 'undislike' : 'dislike', game.id);
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <Link to={`/games/${game.id}`}>
        <div className="h-48 bg-gray-200 flex items-center justify-center">
          <img
            src={`https://cdn.akamai.steamstatic.com/steam/apps/${game.id}/header.jpg`}
            alt={game.title}
            className="w-full h-full object-cover"
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = "https://via.placeholder.com/460x215?text=Game+Image";
            }}
          />
        </div>
      </Link>

      <div className="p-4">
        <div className="flex justify-between mb-2">
          <h3 className="font-bold text-lg truncate">
            <Link to={`/games/${game.id}`} className="hover:text-blue-600">
              {game.title}
            </Link>
          </h3>

          {(showScore || showPopularity) && (
            <div className="flex items-center">
              <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2 py-1 rounded">
                {showScore && game.score && `${(game.score * 100).toFixed(0)}%`}
                {showPopularity && game.popularity && `${(game.popularity * 100).toFixed(0)}%`}
              </span>
            </div>
          )}
        </div>

        {game.tags && game.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {game.tags.slice(0, 3).map((tag, index) => (
              <span key={index} className="bg-gray-200 text-gray-700 text-xs px-2 py-1 rounded">
                {tag}
              </span>
            ))}
          </div>
        )}

        <div className="flex justify-between mt-3">
          <div className="flex space-x-2">
            <button
              onClick={handleLike}
              className={`p-2 rounded ${isLiked ? 'bg-green-100 text-green-700' : 'bg-gray-100 hover:bg-gray-200'}`}
              title={isLiked ? "Unlike" : "Like"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
              </svg>
            </button>

            <button
              onClick={handleDislike}
              className={`p-2 rounded ${isDisliked ? 'bg-red-100 text-red-700' : 'bg-gray-100 hover:bg-gray-200'}`}
              title={isDisliked ? "Remove dislike" : "Dislike"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M18 9.5a1.5 1.5 0 11-3 0v-6a1.5 1.5 0 013 0v6zM14 9.667v-5.43a2 2 0 00-1.105-1.79l-.05-.025A4 4 0 0011.055 2H5.64a2 2 0 00-1.962 1.608l-1.2 6A2 2 0 004.44 12H8v4a2 2 0 002 2 1 1 0 001-1v-.667a4 4 0 01.8-2.4l1.4-1.866a4 4 0 00.8-2.4z" />
              </svg>
            </button>
          </div>

          <Link
            to={`/games/${game.id}`}
            className="text-sm bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded"
          >
            View Details
          </Link>
        </div>
      </div>
    </div>
  );
};

export default GameCard;

