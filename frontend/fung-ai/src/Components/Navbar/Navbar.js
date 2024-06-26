import React from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Navbar({ user, onLogout }) {
  const navigate = useNavigate();

  const logout = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/logout/");
      if (response.status === 200) {
        onLogout();  // Clear local session state
        navigate('/login');  // Navigate to the login page
      } else {
        throw new Error('Failed to logout');
      }
    } catch (error) {
      console.error('Logout error:', error);
      alert('Logout failed, please try again.');  // Provide feedback on failure
    }
  };

  return (
    <nav className="navbar navbar-expand-lg navbar-light bg-light">
      <div className="container-fluid">
        <a className="navbar-brand p-3" href="/">Fung A.I.</a>
        {user && (
          <div className="navbar-text">
            {/* <h5>Hello, {user}</h5> */}
            {user && (
            <button onClick={logout} className="btn me-2">Logout</button>
            )}
          </div>
        )}
      </div>
    </nav>
  );
}

export default Navbar;
