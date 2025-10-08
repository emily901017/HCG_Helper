import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import StudentChat from './components/StudentChat';
import TeacherDashboard from './components/TeacherDashboard';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="nav-logo">📚 HGC Helper</h1>
            <ul className="nav-menu">
              <li className="nav-item">
                <Link to="/" className="nav-link">Student Chat</Link>
              </li>
              <li className="nav-item">
                <Link to="/teacher" className="nav-link">Teacher Dashboard</Link>
              </li>
            </ul>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<StudentChat />} />
          <Route path="/teacher" element={<TeacherDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
