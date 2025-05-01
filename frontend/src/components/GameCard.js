// frontend/src/components/GameCard.js
import React from 'react';
import { Link } from 'react-router-dom';

const GameCard = ({ game, onAction, userActions, showScore }) => {
  // Check if game is in user's action lists
  const isLiked = userActions?.liked?.includes(game.id);
  const isPurchased = userActions?.purchased?.includes(game.id);
  const isRecommended = userActions?.recommended?.includes(game.id);

  // Handle game actions
  const handleLike = (e) => {
    e.preventDefault();
    e.stopPropagation();
    onAction(game.id, isLiked ? 'unlike' : 'like');
  };

  const handleBuy = (e) => {
    e.preventDefault();
    e.stopPropagation();
    onAction(game.id, isPurchased ? 'unbuy' : 'buy');
  };

  const handleRecommend = (e) => {
    e.preventDefault();
    e.stopPropagation();
    onAction(game.id, isRecommended ? 'unrecommend' : 'recommend');
  };

  const cardStyle = {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    overflow: 'hidden',
    transition: 'box-shadow 0.3s',
    display: 'flex',
    flexDirection: 'column',
    height: '100%'
  };

  const imageContainerStyle = {
    height: '160px',
    backgroundColor: '#e5e7eb',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden'
  };

  const imageStyle = {
    width: '100%',
    height: '100%',
    objectFit: 'cover'
  };

  const contentStyle = {
    padding: '1rem',
    flexGrow: 1,
    display: 'flex',
    flexDirection: 'column'
  };

  const titleStyle = {
    fontWeight: 'bold',
    fontSize: '1.125rem',
    marginBottom: '0.5rem',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
  };

  const titleLinkStyle = {
    color: '#1f2937',
    textDecoration: 'none'
  };

  const tagsContainerStyle = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.25rem',
    marginBottom: '0.75rem'
  };

  const tagStyle = {
    backgroundColor: '#e5e7eb',
    color: '#4b5563',
    fontSize: '0.75rem',
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem'
  };

  const priceStyle = {
    marginBottom: '0.75rem',
    color: game.price_final === 0 ? '#059669' : '#4b5563',
    fontWeight: game.price_final === 0 ? 'bold' : 'normal'
  };

  const actionsStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    marginTop: 'auto',
    paddingTop: '0.75rem'
  };

  const buttonGroupStyle = {
    display: 'flex',
    gap: '0.25rem'
  };

  const actionButtonStyle = {
    padding: '0.5rem',
    borderRadius: '9999px',
    border: 'none',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  };

  const likeButtonStyle = {
    ...actionButtonStyle,
    backgroundColor: isLiked ? '#fee2e2' : '#f3f4f6',
    color: isLiked ? '#dc2626' : '#6b7280'
  };

  const buyButtonStyle = {
    ...actionButtonStyle,
    backgroundColor: isPurchased ? '#d1fae5' : '#f3f4f6',
    color: isPurchased ? '#059669' : '#6b7280'
  };

  const recommendButtonStyle = {
    ...actionButtonStyle,
    backgroundColor: isRecommended ? '#dbeafe' : '#f3f4f6',
    color: isRecommended ? '#2563eb' : '#6b7280'
  };

  const detailsButtonStyle = {
    backgroundColor: '#2563eb',
    color: 'white',
    padding: '0.25rem 0.75rem',
    borderRadius: '0.25rem',
    fontSize: '0.875rem',
    textDecoration: 'none',
    border: 'none'
  };

  const scoreStyle = {
    backgroundColor: '#dbeafe',
    color: '#1e40af',
    fontSize: '0.75rem',
    fontWeight: 'bold',
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem',
    marginLeft: '0.5rem'
  };

  return (
    <div style={cardStyle}>
      <Link to={`/games/${game.id}`} style={{ textDecoration: 'none' }}>
        <div style={imageContainerStyle}>
          <img
            src={`https://cdn.akamai.steamstatic.com/steam/apps/${game.id}/header.jpg`}
            alt={game.title}
            style={imageStyle}
            onError={(e) => {
              e.target.onerror = null;
              e.target.src = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='460' height='215' viewBox='0 0 460 215'%3E%3Crect width='460' height='215' fill='%23cccccc'/%3E%3Ctext x='50%25' y='50%25' font-family='Arial' font-size='20' text-anchor='middle' fill='%23666666' dominant-baseline='middle'%3E${encodeURIComponent(game.title || "Game Image")}%3C/text%3E%3C/svg%3E`;
            }}
          />
        </div>
      </Link>

      <div style={contentStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <h3 style={titleStyle}>
            <Link to={`/games/${game.id}`} style={titleLinkStyle}>
              {game.title}
            </Link>
          </h3>

          {showScore && game.score !== undefined && (
            <span style={scoreStyle}>
              {(game.score * 100).toFixed(0)}%
            </span>
          )}
        </div>

        {game.tags && game.tags.length > 0 && (
          <div style={tagsContainerStyle}>
            {game.tags.slice(0, 3).map((tag, index) => (
              <span key={index} style={tagStyle}>
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Price information */}
        {game.price_final !== undefined && (
          <div style={priceStyle}>
            {game.price_final === 0 ? "Free to Play" : `$${game.price_final.toFixed(2)}`}
          </div>
        )}

        {/* Game action buttons */}
        <div style={actionsStyle}>
          <div style={buttonGroupStyle}>
            <button
              onClick={handleLike}
              style={likeButtonStyle}
              title={isLiked ? "Unlike" : "Like"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
              </svg>
            </button>

            <button
              onClick={handleBuy}
              style={buyButtonStyle}
              title={isPurchased ? "Remove from Library" : "Add to Library"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3zM16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
              </svg>
            </button>

            <button
              onClick={handleRecommend}
              style={recommendButtonStyle}
              title={isRecommended ? "Remove Recommendation" : "Recommend"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
              </svg>
            </button>
          </div>

          <Link
            to={`/games/${game.id}`}
            style={detailsButtonStyle}
          >
            Details
          </Link>
        </div>
      </div>
    </div>
  );
};

export default GameCard;