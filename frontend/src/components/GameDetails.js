// frontend/src/components/GameDetails.js
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-toastify';
import GameCard from './GameCard';

const GameDetails = () => {
  const { gameId } = useParams();
  const [game, setGame] = useState(null);
  const [similarGames, setSimilarGames] = useState([]);
  const [loading, setLoading] = useState(true);

  // 从localStorage加载用户操作
  const [userActions, setUserActions] = useState(() => {
    const savedActions = localStorage.getItem('userGameActions');
    return savedActions ? JSON.parse(savedActions) : {
      liked: [],
      purchased: [],
      recommended: []
    };
  });

  // 检查游戏是否在用户操作列表中 - 确保使用数字类型比较
  const isLiked = userActions?.liked?.includes(parseInt(gameId));
  const isPurchased = userActions?.purchased?.includes(parseInt(gameId));
  const isRecommended = userActions?.recommended?.includes(parseInt(gameId));

  useEffect(() => {
    const fetchGameData = async () => {
      setLoading(true);

      try {
        // 并行获取游戏详情和相似游戏
        const [gameResponse, similarGamesResponse] = await Promise.all([
          axios.get(`/api/games/${gameId}`),
          axios.get(`/api/similar-games/${gameId}?count=4`)
        ]);

        if (gameResponse.data.status === 'success') {
          setGame(gameResponse.data.game);
        } else {
          toast.error('加载游戏详情失败');
        }

        setSimilarGames(similarGamesResponse.data.similar_games || []);
      } catch (error) {
        console.error('获取游戏数据时出错:', error);
        toast.error('加载游戏数据失败。请稍后再试。');
      } finally {
        setLoading(false);
      }
    };

    fetchGameData();
  }, [gameId]);

  // 用户操作变更时保存到localStorage
  useEffect(() => {
    localStorage.setItem('userGameActions', JSON.stringify(userActions));
  }, [userActions]);

  const handleGameAction = (gameId, actionType) => {
    setUserActions(prev => {
      // 创建操作数组的副本
      const newActions = { ...prev };
      const gameIdNum = parseInt(gameId);

      // 根据操作类型更新适当的数组
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

    // 显示操作通知
    const actionMessages = {
      like: '游戏已添加到喜欢的游戏',
      buy: '游戏已添加到您的库',
      recommend: '游戏已添加到推荐游戏',
      unlike: '游戏已从喜欢的游戏中移除',
      unbuy: '游戏已从您的库中移除',
      unrecommend: '游戏已从推荐游戏中移除'
    };

    toast.success(actionMessages[actionType] || '操作已记录');
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
        <strong className="font-bold">错误：</strong>
        <span className="block sm:inline"> 未找到游戏或发生错误。</span>
        <div className="mt-4">
          <Link to="/games" className="text-blue-600 hover:text-blue-800">
            &larr; 返回游戏
          </Link>
        </div>
      </div>
    );
  }

  // 处理游戏标签（确保它们是数组格式）
  const gameTags = typeof game.tags === 'string' ? game.tags.split(',').map(tag => tag.trim()) : (game.tags || []);

  return (
    <div className="game-details">
      <div className="mb-4">
        <Link to="/games" className="text-blue-600 hover:text-blue-800 flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
          </svg>
          返回游戏
        </Link>
      </div>

      {/* 游戏头部 */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-8">
        <div className="relative h-64 md:h-80 bg-gray-800">
          <img
            src={`https://cdn.akamai.steamstatic.com/steam/apps/${game.app_id}/header.jpg`}
            alt={game.title}
            className="w-full h-full object-cover"
            onError={(e) => {
              // 尝试备用图片
              e.target.onerror = null;
              // 尝试另一个图片格式
              e.target.src = `https://cdn.akamai.steamstatic.com/steam/apps/${game.app_id}/capsule_616x353.jpg`;

              // 如果还是失败，使用SVG占位符
              e.target.onerror = () => {
                e.target.onerror = null;
                const encodedTitle = encodeURIComponent(game.title || "Game");
                e.target.src = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1200' height='400' viewBox='0 0 1200 400'%3E%3Crect width='1200' height='400' fill='%23444444'/%3E%3Ctext x='50%25' y='50%25' font-family='Arial' font-size='40' text-anchor='middle' fill='%23cccccc' dominant-baseline='middle'%3E${encodedTitle}%3C/text%3E%3C/svg%3E`;
              };
            }}
          />

          {/* 价格标签覆盖层 */}
          {game.price_final !== undefined && (
            <div className="absolute top-4 right-4 bg-black bg-opacity-75 text-white px-4 py-2 rounded-md font-bold">
              {game.price_final === 0 ? "免费游玩" : `$${game.price_final.toFixed(2)}`}
              {game.price_original > game.price_final && (
                <span className="ml-2 text-gray-400 line-through text-sm">${game.price_original.toFixed(2)}</span>
              )}
            </div>
          )}

          {/* Steam Deck 兼容性标签 */}
          {game.steam_deck && (
            <div className="absolute top-4 left-4 bg-black bg-opacity-75 text-white px-4 py-2 rounded-md font-bold flex items-center">
              <svg className="h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"></path>
              </svg>
              Steam Deck 兼容
            </div>
          )}
        </div>

        <div className="p-6">
          <div className="flex flex-wrap justify-between items-start gap-4">
            <div className="mb-4 md:mb-0 flex-grow">
              <h1 className="text-3xl font-bold mb-2">{game.title}</h1>

              {game.date_release && (
                <p className="text-gray-600 mb-2">
                  发布日期: {new Date(game.date_release).toLocaleDateString()}
                </p>
              )}

              {gameTags.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                  {gameTags.map((tag, index) => (
                    <span key={index} className="bg-gray-200 text-gray-700 px-2 py-1 rounded text-sm">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <button
                onClick={() => handleGameAction(gameId, isLiked ? 'unlike' : 'like')}
                className={`flex items-center justify-center space-x-1 px-4 py-2 rounded-md ${
                  isLiked 
                    ? 'bg-red-600 text-white' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
                </svg>
                <span>{isLiked ? '已喜欢' : '喜欢'}</span>
              </button>

              <button
                onClick={() => handleGameAction(gameId, isPurchased ? 'unbuy' : 'buy')}
                className={`flex items-center justify-center space-x-1 px-4 py-2 rounded-md ${
                  isPurchased 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3zM16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
                </svg>
                <span>{isPurchased ? '在库中' : '添加到库'}</span>
              </button>

              <button
                onClick={() => handleGameAction(gameId, isRecommended ? 'unrecommend' : 'recommend')}
                className={`flex items-center justify-center space-x-1 px-4 py-2 rounded-md ${
                  isRecommended 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
                </svg>
                <span>{isRecommended ? '已推荐' : '推荐'}</span>
              </button>
            </div>
          </div>

          {/* 游戏统计数据 */}
          {(game.positive_ratio !== undefined || game.rating !== undefined || game.win !== undefined) && (
            <div className="mt-6 p-4 bg-gray-100 rounded-md">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {game.positive_ratio !== undefined && (
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{(game.positive_ratio * 100).toFixed(0)}%</div>
                    <div className="text-sm text-gray-600">好评率</div>
                  </div>
                )}

                {game.rating !== undefined && (
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{
                      typeof game.rating === 'string' ? game.rating : game.rating.toFixed(1)
                    }</div>
                    <div className="text-sm text-gray-600">评分</div>
                  </div>
                )}

                <div className="text-center col-span-1 sm:col-span-2">
                  <div className="flex justify-center space-x-4">
                    {game.win && (
                      <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full flex items-center">
                        <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M4 3h12v2H4V3zm0 4h12v2H4V7zm0 4h12v2H4v-2zm0 4h12v2H4v-2z"/>
                        </svg>
                        Windows
                      </span>
                    )}
                    {game.mac && (
                      <span className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full flex items-center">
                        <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M17.25 13.25v-10a.5.5 0 00-.5-.5h-7.5a.5.5 0 00-.5.5v10h-1.5v-10a.5.5 0 00-.5-.5h-5.5a.5.5 0 00-.5.5v10H.25v1.5h19.5v-1.5h-.5zm-9.5 0v-9h1.5v9h-1.5zm-7 0v-9h1.5v9h-1.5z"/>
                        </svg>
                        Mac
                      </span>
                    )}
                    {game.linux && (
                      <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full flex items-center">
                        <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M10 .5c-5.247 0-9.5 4.253-9.5 9.5S4.753 19.5 10 19.5s9.5-4.253 9.5-9.5S15.247.5 10 .5zm0 17.5c-4.418 0-8-3.582-8-8s3.582-8 8-8 8 3.582 8 8-3.582 8-8 8z"/>
                        </svg>
                        Linux
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 描述 */}
          {game.description && (
            <div className="mt-6">
              <h3 className="text-xl font-semibold mb-3">关于这款游戏</h3>
              <p className="text-gray-700 leading-relaxed">{game.description}</p>
            </div>
          )}
        </div>
      </div>

      {/* 相似游戏 */}
      {similarGames.length > 0 && (
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">相似游戏</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {similarGames.map(similarGame => (
              <GameCard
                key={similarGame.app_id}
                game={similarGame}
                onAction={handleGameAction}
                userActions={userActions}
                showScore={true}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GameDetails;