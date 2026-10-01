import React, { useState, useEffect } from 'react';
import './App.css';
import CausalAnalysisPanel from './components/CausalAnalysisPanel';

import ResultsVisualization from './components/ResultsVisualization';
import NodeExplorer from './components/NodeExplorer';
import { causalAPI } from './services/api';

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState('disconnected');
  const [nodeData, setNodeData] = useState(null);

  useEffect(() => {
    // Check backend connection on startup
    checkBackendConnection();
    loadNodeData();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const status = await causalAPI.getStatus();
      setModelStatus(status.status);
    } catch (error) {
      console.error('Backend connection failed:', error);
      setModelStatus('error');
    }
  };

  const loadNodeData = async () => {
    try {
      const data = await causalAPI.getNodeData();
      setNodeData(data);
    } catch (error) {
      console.error('Failed to load node data:', error);
    }
  };

  const handleCausalQuery = async (queryData) => {
    setLoading(true);
    try {
      let results;
      if (queryData.analysis_type === 'neuro-symbolic') {
        results = await causalAPI.neuroSymbolicAnalysis(queryData.treatment_idx, queryData.outcome_idx);
      } else {
        results = await causalAPI.performCausalAnalysis(queryData);
      }
      setAnalysisResults(results);
    } catch (error) {
      console.error('Causal analysis failed:', error);
      setAnalysisResults({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>Causal Reasoning Engine for Biological Hypothesis Generation</h1>
          <p className="app-subtitle">Knowledge Graph-Driven Neuro-Symbolic AI for Drug Discovery & Biomedical Research</p>
        </div>
        <div className={`status-indicator ${modelStatus}`}>
          <span className="status-dot"></span>
          Model: {modelStatus}
        </div>
      </header>

      <div className="app-content">
        <div className="left-panel">
          <CausalAnalysisPanel 
            onQuery={handleCausalQuery}
            loading={loading}
            nodeData={nodeData}
          />
          {/* <NodeExplorer nodeData={nodeData} /> */}
        </div>

        <div className="main-panel">
          <ResultsVisualization 
            results={analysisResults}
            loading={loading}
          />
        </div>
      </div>
    </div>
  );
}

export default App;