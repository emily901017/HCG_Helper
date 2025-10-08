import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './TeacherDashboard.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function TeacherDashboard() {
  const [queries, setQueries] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [keywords, setKeywords] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [activeTab, setActiveTab] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [queriesRes, statsRes, keywordsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/teacher/queries`),
        axios.get(`${API_BASE_URL}/api/teacher/statistics`),
        axios.get(`${API_BASE_URL}/api/teacher/keywords?limit=20`)
      ]);

      setQueries(queriesRes.data.queries);
      setStatistics(statsRes.data);
      setKeywords(keywordsRes.data.keywords);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchTerm.trim()) return;

    try {
      const response = await axios.post(`${API_BASE_URL}/api/teacher/search`, {
        keyword: searchTerm
      });
      setSearchResults(response.data.results);
    } catch (error) {
      console.error('Error searching:', error);
    }
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getSubjectChartData = () => {
    if (!statistics || !statistics.by_subject) return [];
    return Object.entries(statistics.by_subject).map(([subject, count]) => ({
      subject,
      count
    }));
  };

  const getTimelineData = () => {
    const dateMap = {};
    queries.forEach(query => {
      const date = new Date(query.timestamp).toLocaleDateString();
      dateMap[date] = (dateMap[date] || 0) + 1;
    });

    return Object.entries(dateMap)
      .sort((a, b) => new Date(a[0]) - new Date(b[0]))
      .map(([date, count]) => ({ date, count }));
  };

  const downloadCSV = () => {
    const csvContent = [
      ['Timestamp', 'Question', 'Session ID'],
      ...queries.map(q => [
        formatDate(q.timestamp),
        `"${q.question.replace(/"/g, '""')}"`,
        q.session_id
      ])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `student_queries_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h2>Teacher Dashboard</h2>
        <p>Analyze student questions to identify learning difficulties</p>
        <button onClick={fetchData} className="refresh-button">
          🔄 Refresh Data
        </button>
      </div>

      {/* Statistics Cards */}
      <div className="stats-cards">
        <div className="stat-card">
          <h3>Total Questions</h3>
          <p className="stat-number">{statistics?.total_queries || 0}</p>
        </div>
        <div className="stat-card">
          <h3>Last 7 Days</h3>
          <p className="stat-number">{statistics?.recent_queries_7days || 0}</p>
        </div>
        <div className="stat-card">
          <h3>Unique Sessions</h3>
          <p className="stat-number">
            {queries.length > 0 ? new Set(queries.map(q => q.session_id)).size : 0}
          </p>
        </div>
      </div>

      {/* Subject Distribution Chart */}
      {statistics?.by_subject && Object.keys(statistics.by_subject).length > 0 && (
        <div className="chart-section">
          <h3>Questions by Subject</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={getSubjectChartData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="subject" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#667eea" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >
          All Questions
        </button>
        <button
          className={`tab ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          Search
        </button>
        <button
          className={`tab ${activeTab === 'insights' ? 'active' : ''}`}
          onClick={() => setActiveTab('insights')}
        >
          Insights
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'all' && (
          <div className="all-queries-tab">
            <div className="queries-header">
              <h3>All Student Questions</h3>
              {queries.length > 0 && (
                <button onClick={downloadCSV} className="download-button">
                  📥 Download CSV
                </button>
              )}
            </div>

            {queries.length === 0 ? (
              <p className="no-data">No questions logged yet.</p>
            ) : (
              <div className="queries-table">
                <table>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Question</th>
                      <th>Session ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {queries.map((query, index) => (
                      <tr key={index}>
                        <td>{formatDate(query.timestamp)}</td>
                        <td>{query.question}</td>
                        <td className="session-id">{query.session_id?.substring(0, 8)}...</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {activeTab === 'search' && (
          <div className="search-tab">
            <h3>Search Questions</h3>
            <form onSubmit={handleSearch} className="search-form">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Enter keyword to search..."
                className="search-input"
              />
              <button type="submit" className="search-button">
                Search
              </button>
            </form>

            {searchResults.length > 0 && (
              <div className="search-results">
                <p className="results-count">Found {searchResults.length} matching questions</p>
                <div className="queries-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Question</th>
                      </tr>
                    </thead>
                    <tbody>
                      {searchResults.map((result, index) => (
                        <tr key={index}>
                          <td>{formatDate(result.timestamp)}</td>
                          <td>{result.question}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'insights' && (
          <div className="insights-tab">
            <h3>Question Insights</h3>

            {/* Keywords Section */}
            <div className="keywords-section">
              <h4>Most Common Keywords</h4>
              {keywords.length > 0 ? (
                <div className="keywords-grid">
                  {keywords.map(([keyword, count], index) => (
                    <div key={index} className="keyword-item">
                      <span className="keyword">{keyword}</span>
                      <span className="count">{count}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-data">No keyword data available.</p>
              )}
            </div>

            {/* Timeline Chart */}
            {queries.length > 0 && (
              <div className="timeline-section">
                <h4>Question Frequency Over Time</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={getTimelineData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="count" stroke="#667eea" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default TeacherDashboard;
