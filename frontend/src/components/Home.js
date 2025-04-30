
// frontend/src/components/Home.js
import React from 'react';
import { Link } from 'react-router-dom';
import GameCard from './GameCard';

const Home = ({ user, recommendations, popularGames, userPreferences, updatePreference }) => {
  return (
    <div className="home">
      {/* Hero section */}
      <section className="bg-gradient-to-r from-blue-800 to-purple-800 text-white py-12 px-4 rounded-lg mb-8">
        <div className="container mx-auto">
          <h1 className="text-4xl font-bold mb-4">Discover Your Next Favorite Game</h1>
          <p className="text-xl mb-6">
            Personalized game recommendations based on your preferences and playing history
          </p>

          {!user && (
            <div className="flex space-x-4">
              <Link
                to="/register"
                className="bg-white text-blue-800 hover:bg-gray-100 px-6 py-3 rounded-md font-semibold"
              >
                Sign Up Now
              </Link>
              <Link
                to="/login"
                className="border border-white text-white hover:bg-white hover:text-blue-800 px-6 py-3 rounded-md font-semibold"
              >
                Login
              </Link>
            </div>
          )}
        </div>
      </section>

      {/* Personalized recommendations section */}
      {user && recommendations.length > 0 && (
        <section className="mb-12">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">Your Personalized Recommendations</h2>
            <Link to="/profile" className="text-blue-600 hover:text-blue-800">View Profile</Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {recommendations.slice(0, 8).map(game => (
              <GameCard
                key={game.id}
                game={game}
                userPreferences={userPreferences}
                updatePreference={updatePreference}
                showScore={true}
              />
            ))}
          </div>
        </section>
      )}

      {/* Popular games section */}
      <section>
        <h2 className="text-2xl font-bold mb-6">Popular Games</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {popularGames.map(game => (
            <GameCard
              key={game.id}
              game={game}
              userPreferences={userPreferences}
              updatePreference={updatePreference}
              showPopularity={true}
            />
          ))}
        </div>
      </section>
    </div>
  );
};

export default Home;

