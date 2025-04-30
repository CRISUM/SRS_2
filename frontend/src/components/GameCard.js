// frontend/src/components/GameCard.js
import React from 'react';
import { Link } from 'react-router-dom';

const GameCard = ({ game, onAction, userActions, showScore }) => {
  // 检查游戏在用户各操作列表中的状态
  const isLiked = userActions?.liked?.includes(game.id);
  const isPurchased = userActions?.purchased?.includes(game.id);
  const isRecommended = userActions?.recommended?.includes(game.id);

  // 处理游戏操作
  const handleLike = () => {
    onAction(game.id, isLiked ? 'unlike' : 'like');
  };

  const handleBuy = () => {
    onAction(game.id, isPurchased ? 'unbuy' : 'buy');
  };

  const handleRecommend = () => {
    onAction(game.id, isRecommended ? 'unrecommend' : 'recommend');
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300">
      <Link to={`/games/${game.id}`}>
        <div className="h-48 bg-gray-200 flex items-center justify-center overflow-hidden">
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

          {showScore && game.score && (
            <div className="flex items-center">
              <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2 py-1 rounded">
                {(game.score * 100).toFixed(0)}%
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

        {/* 价格信息 */}
        {game.price_final !== undefined && (
          <div className="mb-3">
            <span className={game.price_final === 0 ? "text-green-600 font-bold" : "text-gray-700"}>
              {game.price_final === 0 ? "Free to Play" : `$${game.price_final.toFixed(2)}`}
            </span>
          </div>
        )}

        {/* 游戏操作按钮 */}
        <div className="flex justify-between mt-3">
          <div className="flex space-x-1">
            <button
              onClick={handleLike}
              className={`p-2 rounded-full ${isLiked ? 'bg-red-100 text-red-600' : 'bg-gray-100 hover:bg-gray-200 text-gray-600'}`}
              title={isLiked ? "Unlike" : "Like"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
              </svg>
            </button>

            <button
              onClick={handleBuy}
              className={`p-2 rounded-full ${isPurchased ? 'bg-green-100 text-green-600' : 'bg-gray-100 hover:bg-gray-200 text-gray-600'}`}
              title={isPurchased ? "Remove from Library" : "Add to Library"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3zM16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
              </svg>
            </button>

            <button
              onClick={handleRecommend}
              className={`p-2 rounded-full ${isRecommended ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 hover:bg-gray-200 text-gray-600'}`}
              title={isRecommended ? "Remove Recommendation" : "Recommend"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
              </svg>
            </button>
          </div>

          <Link
            to={`/games/${game.id}`}
            className="text-sm bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded"
          >
            Details
          </Link>
        </div>
      </div>
    </div>
  );
};

export default GameCard;
