import React, { useState } from 'react';
import './Signup.css';
import { Link } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import Cookies from 'js-cookie'; // Import js-cookie to handle CSRF token

const Signup = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',  // Assuming your backend also expects an email, adjust as necessary
    password: ''
  });
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const csrftoken = Cookies.get('csrftoken');  // Fetch the CSRF token from the cookie

    try {
      console.log("Printing signup request body here...")
      console.log(JSON.stringify(formData))
      const response = await axios.post('http://127.0.0.1:8000/register/', JSON.stringify(formData), {
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrftoken  // Include CSRF token in the request header
        }
      });
      if (response.status === 200) {
        alert('Account created successfully! Please log in.');
        navigate('/');
        // Optionally redirect the user to the login page or handle login directly
      } else {
        alert('Failed to create account: ' + response.data.message); // Handling backend validation feedback
      }
    } catch (error) {
      console.error('Signup error:', error);
      alert('Signup failed. Please check the console for more info.'); // Provide generic error message and detailed console log
    }
  };

  return (
    <div className="signup-container">
      <h2>Sign Up</h2>
      <form className="signup-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username">Username</label>
          <input type="text" id="username" name="username" value={formData.username} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input type="email" id="email" name="email" value={formData.email} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input type="password" id="password" name="password" value={formData.password} onChange={handleChange} required />
        </div>
        <button type="submit" className="signup-button">Create Account</button>
      </form>
      <div className="login-link-container">
        <Link to="/" className="login-link">Already have an account? Log in</Link>
      </div>
    </div>
  );
}

export default Signup;
