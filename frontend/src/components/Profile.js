// frontend/src/components/Profile.js
import React from 'react';
import { Link } from 'react-router-dom';
import GameCard from './GameCard';

const Profile = ({ user, userPreferences, updatePreference }) => {
  const likedGames = userPreferences?.liked_games || [];
  const dislikedGames = userPreferences?.disliked_games || [];
  const playedGames = userPreferences?.played_games || [];

  return (
    <div className="profile">
      <h1 className="text-3xl font-bold mb-6">Your Profile</h1>

      <div className="bg-white rounded-lg shadow-md overflow-hidden p-6 mb-8">
        <h2 className="text-xl font-semibold mb-2">User Information</h2>
        <p className="text-gray-600">Username: {user?.username || user?.id}</p>

        <div className="mt-4">
          <p className="text-gray-600">
            <span className="font-semibold">{likedGames.length}</span> liked games
            {' • '}
            <span className="font-semibold">{dislikedGames.length}</span> disliked games
          </p>
        </div>
      </div>

      {/* Liked games section */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold mb-4">Games You Like</h2>

        {likedGames.length === 0 ? (
          <div className="bg-gray-100 p-4 rounded text-gray-600">
            <p>You haven't liked any games yet.</p>
            <p className="mt-2">
              <Link to="/games" className="text-blue-600 hover:text-blue-800">
                Browse games
              </Link>
              {' '}to find something you like!
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {likedGames.map(game => (
              <GameCard
                key={game.id}
                game={game}
                userPreferences={userPreferences}
                updatePreference={updatePreference}
              />
            ))}
          </div>
        )}
      </section>

      {/* Disliked games section */}
      {dislikedGames.length > 0 && (
        <section className="mb-10">
          <h2 className="text-2xl font-bold mb-4">Games You Dislike</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {dislikedGames.map(game => (
              <GameCard
                key={game.id}
                game={game}
                userPreferences={userPreferences}
                updatePreference={updatePreference}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
};

export default Profile;