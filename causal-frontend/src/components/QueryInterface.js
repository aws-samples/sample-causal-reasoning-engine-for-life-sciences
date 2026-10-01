import React, { useState } from 'react';
import { MessageSquare, Send, Loader } from 'lucide-react';

const QueryInterface = ({ onQuery, loading }) => {
  const [question, setQuestion] = useState('');
  const [queryHistory, setQueryHistory] = useState([]);

  const exampleQuestions = [
    "What is the causal effect of aspirin on cardiovascular disease?",
    "How does metformin affect type 2 diabetes?",
    "What would happen if we intervened on APOE gene expression?",
    "Does ibuprofen work through COX2 to reduce inflammation?",
    "What is the mediation pathway from statins to heart disease?",
    "How do confounders affect the warfarin-stroke relationship?"
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    const queryData = {
      analysis_type: 'natural_language',
      question: question.trim()
    };

    setQueryHistory(prev => [...prev, { question: question.trim(), timestamp: new Date() }]);
    onQuery(queryData);
    setQuestion('');
  };

  const handleExampleClick = (example) => {
    setQuestion(example);
  };

  return (
    <div className="query-interface">
      <div className="query-header">
        <MessageSquare size={20} />
        <h3>Ask Causal Questions</h3>
      </div>

      <form onSubmit={handleSubmit} className="query-form">
        <div className="input-group">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a causal question about drug-disease relationships..."
            rows={3}
            disabled={loading}
          />
          <button 
            type="submit" 
            disabled={loading || !question.trim()}
            className="send-btn"
          >
            {loading ? <Loader size={16} className="spinning" /> : <Send size={16} />}
          </button>
        </div>
      </form>

      <div className="example-questions">
        <h4>Example Questions:</h4>
        <div className="examples-grid">
          {exampleQuestions.map((example, idx) => (
            <button
              key={idx}
              className="example-btn"
              onClick={() => handleExampleClick(example)}
              disabled={loading}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {queryHistory.length > 0 && (
        <div className="query-history">
          <h4>Recent Questions:</h4>
          <div className="history-list">
            {queryHistory.slice(-5).reverse().map((item, idx) => (
              <div key={idx} className="history-item">
                <span className="history-question">{item.question}</span>
                <span className="history-time">
                  {item.timestamp.toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default QueryInterface;