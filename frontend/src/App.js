// src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// 组件导入
import Header from './components/Header';
import Login from './components/Login';
import Register from './components/Register';
import Home from './components/Home';
import GameDetails from './components/GameDetails';
import GamesList from './components/GamesList';
import Profile from './components/Profile';
import Footer from './components/Footer';

// API 基础URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Axios 配置
axios.defaults.baseURL = API_BASE_URL;

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState([]);
  const [popularGames, setPopularGames] = useState([]);
  const [userPreferences, setUserPreferences] = useState({
    liked_games: [],
    disliked_games: [],
    played_games: []
  });

  // 初始化 - 检查是否已登录
  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (token && userData) {
      try {
        const parsedUser = JSON.parse(userData);
        setUser(parsedUser);
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        fetchUserData();
      } catch (error) {
        console.error('Error parsing user data:', error);
        logout();
      }
    }
    
    setLoading(false);
    fetchPopularGames();
  }, []);

  // 获取用户数据 (推荐和偏好)
  const fetchUserData = async () => {
    try {
      const [recommendationsResponse, preferencesResponse] = await Promise.all([
        axios.get('/recommendations'),
        axios.get('/user/preferences')
      ]);
      
      setRecommendations(recommendationsResponse.data.recommendations || []);
      setUserPreferences(preferencesResponse.data.preferences || {
        liked_games: [],
        disliked_games: [],
        played_games: []
      });
    } catch (error) {
      console.error('Error fetching user data:', error);
      toast.error('Failed to load your data. Please try again later.');
    }
  };

  // 获取流行游戏
  const fetchPopularGames = async () => {
    try {
      const response = await axios.get('/popular-games?count=10');
      setPopularGames(response.data.popular_games || []);
    } catch (error) {
      console.error('Error fetching popular games:', error);
    }
  };

  // 登录
  const login = async (username, password) => {
    try {
      const response = await axios.post('/login', { username, password });
      const { token, user } = response.data;
      
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(user));
      
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      setUser(user);
      
      fetchUserData();
      toast.success('Login successful!');
      return true;
    } catch (error) {
      console.error('Login error:', error);
      toast.error(error.response?.data?.message || 'Login failed. Please try again.');
      return false;
    }
  };

  // 注册
  const register = async (username, password) => {
    try {
      const response = await axios.post('/register', { username, password });
      const { token, user } = response.data;
      
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(user));
      
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      setUser(user);
      
      toast.success('Registration successful!');
      return true;
    } catch (error) {
      console.error('Registration error:', error);
      toast.error(error.response?.data?.message || 'Registration failed. Please try again.');
      return false;
    }
  };

  // 登出
  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
    setRecommendations([]);
    setUserPreferences({
      liked_games: [],
      disliked_games: [],
      played_games: []
    });
    toast.info('You have been logged out.');
  };

  // 更新用户游戏偏好
  const updatePreference = async (action, gameId) => {
    if (!user) {
      toast.warning('Please login to save your preferences');
      return false;
    }
    
    try {
      await axios.post('/user/preferences', {
        action,
        game_id: gameId
      });
      
      // 更新本地状态
      fetchUserData();
      return true;
    } catch (error) {
      console.error('Error updating preferences:', error);
      toast.error('Failed to update your preferences. Please try again.');
      return false;
    }
  };

  // 检查是否为受保护路由
  const ProtectedRoute = ({ children }) => {
    if (loading) return <div className="loading">Loading...</div>;
    if (!user) return <Navigate to="/login" />;
    return children;
  };

  return (
    <Router>
      <div className="app">
        <Header user={user} onLogout={logout} />
        
        <main className="container mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={
              <Home 
                user={user} 
                recommendations={recommendations} 
                popularGames={popularGames} 
                userPreferences={userPreferences}
                updatePreference={updatePreference}
              />
            } />
            
            <Route path="/login" element={
              user ? <Navigate to="/" /> : <Login onLogin={login} />
            } />
            
            <Route path="/register" element={
              user ? <Navigate to="/" /> : <Register onRegister={register} />
            } />
            
            <Route path="/games" element={<GamesList />} />
            
            <Route path="/games/:gameId" element={
              <GameDetails 
                userPreferences={userPreferences}
                updatePreference={updatePreference} 
              />
            } />
            
            <Route path="/profile" element={
              <ProtectedRoute>
                <Profile 
                  user={user} 
                  userPreferences={userPreferences}
                  updatePreference={updatePreference}
                />
              </ProtectedRoute>
            } />
          </Routes>
        </main>
        
        <Footer />
        <ToastContainer position="bottom-right" autoClose={3000} />
      </div>
    </Router>
  );
}

export default App;