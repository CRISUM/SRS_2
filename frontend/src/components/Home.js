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
    // 从本地存储加载用户操作
    const savedActions = localStorage.getItem('userGameActions');
    return savedActions ? JSON.parse(savedActions) : {
      liked: [],
      purchased: [],
      recommended: []
    };
  });
  const [showRecommendations, setShowRecommendations] = useState(false);

  // 加载随机游戏
  useEffect(() => {
    fetchRandomGames();
  }, []);

  // 保存用户操作到本地存储
  useEffect(() => {
    localStorage.setItem('userGameActions', JSON.stringify(userActions));
  }, [userActions]);

  const fetchRandomGames = async () => {
    setLoading(true);
    try {
      // 从游戏列表API获取更多游戏以创建随机选择
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
      // 创建操作数组的副本
      const newActions = { ...prev };
      
      // 根据操作类型更新相应数组
      switch (actionType) {
        case 'like':
          if (!newActions.liked.includes(gameId)) {
            newActions.liked = [...newActions.liked, gameId];
          }
          break;
        case 'buy':
          if (!newActions.purchased.includes(gameId)) {
            newActions.purchased = [...newActions.purchased, gameId];
          }
          break;
        case 'recommend':
          if (!newActions.recommended.includes(gameId)) {
            newActions.recommended = [...newActions.recommended, gameId];
          }
          break;
        case 'unlike':
          newActions.liked = newActions.liked.filter(id => id !== gameId);
          break;
        case 'unbuy':
          newActions.purchased = newActions.purchased.filter(id => id !== gameId);
          break;
        case 'unrecommend':
          newActions.recommended = newActions.recommended.filter(id => id !== gameId);
          break;
        default:
          break;
      }
      
      return newActions;
    });
    
    // 显示操作的通知
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
      // 发送用户操作历史到后端以获取推荐
      const response = await axios.post('/api/custom-recommendations', userActions);
      setRecommendations(response.data.recommendations || []);
      setShowRecommendations(true);
      
      // 显示成功消息
      toast.success('Recommendations generated based on your preferences!');
    } catch (error) {
      console.error('Error getting recommendations:', error);
      toast.error('Failed to generate recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetActions = () => {
    // 清除所有用户操作
    setUserActions({
      liked: [],
      purchased: [],
      recommended: []
    });
    setShowRecommendations(false);
    toast.info('Your preferences have been reset');
  };

  // 确定要显示的游戏列表
  const displayGames = searchTerm ? searchResults : randomGames;

  return (
    <div className="home">
      {/* 搜索栏 */}
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

      {/* 用户操作计数和操作按钮 */}
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

      {/* 推荐区域 */}
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

      {/* 游戏展示区域 */}
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
