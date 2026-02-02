"use client";

import { useState } from 'react';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  sources?: string[]; // Optional sources for bot messages
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSourceId, setExpandedSourceId] = useState<number | null>(null); // State to manage expanded source

  const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_FASTAPI_BASE_URL || 'http://localhost:8000';

  const handleSendMessage = async () => {
    if (input.trim()) {
      const newUserMessage: Message = { id: messages.length, text: input, sender: 'user' };
      setMessages(prevMessages => [...prevMessages, newUserMessage]);
      setInput('');
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${FASTAPI_BASE_URL}/chat?query=${encodeURIComponent(newUserMessage.text)}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const botResponse: Message = { 
          id: messages.length + 1, 
          text: data.response, 
          sender: 'bot',
          sources: data.sources // Assuming FastAPI returns a 'sources' array
        };
        setMessages(prevMessages => [...prevMessages, botResponse]);

      } catch (e: any) {
        console.error("Error fetching from FastAPI:", e);
        setError(`Failed to get response from chatbot. Error: ${e.message}`);
        setMessages(prevMessages => [...prevMessages, {
          id: prevMessages.length + 1,
          text: `Error: ${e.message}`,
          sender: 'bot'
        }]);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !loading) {
      handleSendMessage();
    }
  };

  const toggleSourceExpansion = (messageId: number) => {
    setExpandedSourceId(expandedSourceId === messageId ? null : messageId);
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      maxWidth: '800px',
      margin: '0 auto',
      border: '1px solid #ccc',
      borderRadius: '8px',
      overflow: 'hidden',
      fontFamily: 'Arial, sans-serif'
    }}>
      <h1 style={{ textAlign: 'center', padding: '10px', borderBottom: '1px solid #eee', margin: 0, backgroundColor: '#f0f2f5' }}>
        iMessage Memory Chatbot
      </h1>
      <div style={{ flexGrow: 1, padding: '15px', overflowY: 'auto', backgroundColor: '#e5ddd5' }}>
        {messages.map(msg => (
          <div 
            key={msg.id} 
            style={{
              marginBottom: '10px',
              padding: '8px 12px',
              borderRadius: '15px',
              maxWidth: '70%',
              backgroundColor: msg.sender === 'user' ? '#dcf8c6' : '#fff',
              marginLeft: msg.sender === 'user' ? 'auto' : '10px',
              marginRight: msg.sender === 'bot' ? 'auto' : '10px',
              boxShadow: '0 1px 0.5px rgba(0, 0, 0, 0.13)',
              wordBreak: 'break-word'
            }}
          >
            <strong>{msg.sender === 'user' ? 'You' : 'Bot'}:</strong> {msg.text}
            {msg.sources && msg.sources.length > 0 && (
              <div style={{ marginTop: '5px', fontSize: '0.8em', color: '#666' }}>
                <strong 
                  style={{ cursor: 'pointer', color: '#007bff' }} 
                  onClick={() => toggleSourceExpansion(msg.id)}
                >
                  Sources ({msg.sources.length}) {expandedSourceId === msg.id ? '▲' : '▼'}
                </strong>
                {expandedSourceId === msg.id && (
                  <ul style={{ listStyleType: 'none', padding: 0, margin: '5px 0 0 0' }}>
                    {msg.sources.map((source, index) => (
                      <li key={index} style={{ marginBottom: '5px', padding: '5px', backgroundColor: '#f0f0f0', borderRadius: '5px' }}>
                        {source}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ 
            marginBottom: '10px',
            padding: '8px 12px',
            borderRadius: '15px',
            maxWidth: '70%',
            backgroundColor: '#fff',
            marginRight: 'auto',
            marginLeft: '10px',
            boxShadow: '0 1px 0.5px rgba(0, 0, 0, 0.13)',
            color: '#555'
          }}>
            <strong>Bot:</strong> Thinking...
          </div>
        )}
        {error && (
          <div style={{ 
            marginBottom: '10px',
            padding: '8px 12px',
            borderRadius: '15px',
            maxWidth: '90%',
            backgroundColor: '#ffe0e0',
            marginLeft: '10px',
            marginRight: 'auto',
            boxShadow: '0 1px 0.5px rgba(0, 0, 0, 0.13)',
            color: '#d32f2f'
          }}>
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', padding: '10px', borderTop: '1px solid #eee', backgroundColor: '#f0f2f5' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={loading ? "Waiting for response..." : "Type your message..."}
          disabled={loading}
          style={{
            flexGrow: 1,
            padding: '10px',
            border: '1px solid #ddd',
            borderRadius: '20px',
            marginRight: '10px',
            outline: 'none',
            backgroundColor: loading ? '#e0e0e0' : 'white'
          }}
        />
        <button
          onClick={handleSendMessage}
          disabled={loading || !input.trim()}
          style={{
            padding: '10px 20px',
            backgroundColor: (loading || !input.trim()) ? '#a0c7ff' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '20px',
            cursor: (loading || !input.trim()) ? 'not-allowed' : 'pointer',
            outline: 'none',
            transition: 'background-color 0.2s'
          }}
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}