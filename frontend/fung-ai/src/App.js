import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './Components/Navbar/Navbar';
import Login from './Components/Login/Login';
import MyDropzone from './Components/MyDropZone/MyDropZone';
import Signup from './Components/Signup/Signup';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import Canvas from './ImageComponent';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState('');

  useEffect(() => {
    const storedIsLoggedIn = sessionStorage.getItem('isLoggedIn');
    const storedUser = sessionStorage.getItem('user');

    if (storedIsLoggedIn === 'true' && storedUser) {
      setIsLoggedIn(true);
      setUser(storedUser);
    }
  }, []);

  const handleLoginSuccess = (isLoggedIn, username) => {
    if (isLoggedIn) {
      console.log(username);
      setIsLoggedIn(true);
      setUser(username);
      sessionStorage.setItem('isLoggedIn', 'true');
      sessionStorage.setItem('user', username);
    } else {
      alert('Invalid username or password');
    }
  };

  const handleLogout = () => {
    console.log('Logging out');
    setIsLoggedIn(false);
    setUser('');
    sessionStorage.removeItem('isLoggedIn');
    sessionStorage.removeItem('user');
    console.log('User logged out');
    setTimeout(() => {
      window.location.href = '/';
    }, 0);
  };

  return (
    <Router>
      <div className="App">
        <Navbar user={user} onLogout={handleLogout} />
        {isLoggedIn ? (
          <>
            <MyDropzone user={user} />
          </>
        ) : (
          <Routes>
            <Route path="/" element={<Login onLogin={handleLoginSuccess} />} />
            <Route path="/login" element={<Login onLogin={handleLoginSuccess} />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="*" element={<Navigate to="/login" replace />} />
          </Routes>
        )}
        
      </div>
    </Router>
  );
}

export default App;
