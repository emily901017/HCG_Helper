import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './StudentChat.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:6000';

function StudentChat() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [expandedSources, setExpandedSources] = useState({});
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Generate session ID on mount
    setSessionId(generateSessionId());
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages change
    scrollToBottom();
  }, [messages]);

  const generateSessionId = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : ((r & 0x3) | 0x8);
      return v.toString(16);
    });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!inputValue.trim() || loading) return;

    const userMessage = inputValue.trim();
    setInputValue('');

    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat/query`, {
        question: userMessage,
        session_id: sessionId
      });

      // Add assistant response with sources to chat
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources || []
      }]);

    } catch (error) {
      console.error('Error fetching response:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        sources: []
      }]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(generateSessionId());
    setExpandedSources({});
  };

  const toggleSources = (messageIndex) => {
    setExpandedSources(prev => ({
      ...prev,
      [messageIndex]: !prev[messageIndex]
    }));
  };

  return (
    <div className="student-chat-container">
      <div className="chat-main">
        <div className="chat-header">
          <h2>Your Learning Companion</h2>
          <p>Ask me anything about History, Geography, or Civics!</p>
        </div>

        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <h3>👋 Welcome to HGC Helper!</h3>
              <p>Start by asking a question about your social studies subjects.</p>
            </div>
          )}

          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="message-avatar">
                {message.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-wrapper">
                <div className="message-content">
                  {message.content}
                </div>
                {message.sources && message.sources.length > 0 && (
                  <div className="message-sources">
                    <button
                      className="sources-toggle"
                      onClick={() => toggleSources(index)}
                    >
                      <span className="toggle-icon">
                        {expandedSources[index] ? '▼' : '▶'}
                      </span>
                      📚 Sources ({message.sources.length})
                    </button>
                    {expandedSources[index] && (
                      <div className="sources-list">
                        {message.sources.map((source) => (
                          <div key={source.index} className="source-item">
                            <div className="source-badge">{source.index}</div>
                            <div className="source-details">
                              <div className="source-meta">
                                <span className="source-subject">{source.subject}</span>
                                <span className="source-level">{source.level}</span>
                              </div>
                              <div className="source-preview">{source.text_preview}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content loading">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question..."
            className="chat-input"
            disabled={loading}
          />
          <button
            type="submit"
            className="send-button"
            disabled={loading || !inputValue.trim()}
          >
            Send
          </button>
        </form>
      </div>

      <div className="chat-sidebar">
        <div className="sidebar-section">
          <h3>About</h3>
          <p>
            HGC Helper is an AI-powered tutor for high school social studies.
            Ask questions about History, Geography, and Civics, and get accurate
            answers based on your textbooks.
          </p>
        </div>

        <div className="sidebar-section">
          <h3>Tips for Better Questions</h3>
          <ul>
            <li>Be specific about what you want to know</li>
            <li>Mention the subject or topic if relevant</li>
            <li>Ask one question at a time</li>
            <li>Use proper terminology when possible</li>
          </ul>
        </div>

        <button onClick={clearChat} className="clear-button">
          Clear Chat History
        </button>
      </div>
    </div>
  );
}

export default StudentChat;
