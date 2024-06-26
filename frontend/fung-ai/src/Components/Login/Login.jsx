import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Login.css';
import axios from 'axios';  // Import Axios to make HTTP requests

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleUsernameChange = (event) => {
    setUsername(event.target.value);
  };

  const handlePasswordChange = (event) => {
    setPassword(event.target.value);
  };

  const login = async () => {
    if (username && password) {
      const loginUrl = 'http://127.0.0.1:8000/login/'; // Replace with your actual backend URL
    
      try {
        // const response = await axios.post(loginUrl, {
        //   username: username,
        //   password: password
        // });
        const response = await axios.post(loginUrl, {
          username: username,
          password: password
        }, {
          headers: {
            'Content-Type': 'application/json'
          }
        });      
        if (response.status===200) {
          onLogin(true, username); 
          window.location.href = '/';
          console.log('Login successful: ' + response.data.message);
        } else {
          alert('Login failed: ' + response.data.message); // Assuming the backend sends back a message
        }
      } catch (error) {
        console.error('Login error:', error);
        alert('Login failed. Please check the console for more info.');
      }
    } else {
      alert('Please enter both username and password');
    }
  };

  return (
    <div className='container'>
      <div className="header">
        <div className="text">Login</div>
        <div className="underline"></div>
      </div>
      <div className="Username">
        <input type="text" value={username} onChange={handleUsernameChange} placeholder="Username" />
      </div>
      <div className="Password">
        <input type="password" value={password} onChange={handlePasswordChange} placeholder="Password" />
      </div>
      {/* <div className="Forgot-Password">Forgot Password</div> */}
      <div className="submit_container">
        <button onClick={login} className="login">Login</button>
      </div>
      {/* <div className="signup_link">
        <Link to="/signup">Don't have an account? Sign up</Link>
      </div> */}
    </div>
  );
}

export default Login;
