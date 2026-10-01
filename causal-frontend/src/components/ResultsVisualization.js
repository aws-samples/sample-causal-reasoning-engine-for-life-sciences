import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { TrendingUp, AlertCircle, CheckCircle, Info, Zap, GitBranch, Activity, MessageSquare } from 'lucide-react';
import KnowledgeGraphVisualization from './KnowledgeGraphVisualization';
import { causalAPI } from '../services/api';

const ResultsVisualization = ({ results, loading }) => {
  const [showGraph, setShowGraph] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [loadingGraph, setLoadingGraph] = useState(false);

  useEffect(() => {
    if (results && showGraph) {
      loadGraphData();
    }
  }, [results, showGraph]);

  const loadGraphData = async () => {
    if (!results) return;
    
    setLoadingGraph(true);
    try {
      // Extract treatment and outcome indices from results
      let treatmentIdx = 0;
      let outcomeIdx = 0;
      
      // Try to find indices from different result types
      if (results.interventional || results.backdoor || results.counterfactual || results.mediation || results.neuro_symbolic) {
        // For now, use default indices - in a real implementation, 
        // you'd pass these from the analysis request
        const graphResponse = await causalAPI.getSubgraph(treatmentIdx, outcomeIdx);
        setGraphData(graphResponse);
      }
    } catch (error) {
      console.error('Failed to load graph data:', error);
    } finally {
      setLoadingGraph(false);
    }
  };
  if (loading) {
    return (
      <div className="results-loading">
        <div className="loading-spinner"></div>
        <p>Running causal analysis...</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-placeholder">
        <Info size={48} />
        <h3>No Analysis Results</h3>
        <p>Run a causal analysis to see results here</p>
      </div>
    );
  }

  if (results.error) {
    return (
      <div className="results-error">
        <AlertCircle size={48} />
        <h3>Analysis Error</h3>
        <p>{results.error}</p>
      </div>
    );
  }

  const renderInterventionalResults = () => {
    if (!results.interventional) return null;
    
    return (
      <div className="result-section">
        <h4><Zap size={16} /> Interventional Prediction</h4>
        <div className="metric-card">
          <div className="metric-value">{results.interventional.probability?.toFixed(3)}</div>
          <div className="metric-label">P(Y|do(X))</div>
          <div className="metric-uncertainty">±{results.interventional.uncertainty?.toFixed(3)}</div>
        </div>
        <p className="interpretation">
          The causal effect of the intervention has a {(results.interventional.probability * 100).toFixed(1)}% 
          probability of success.
        </p>
      </div>
    );
  };

  const renderCounterfactualResults = () => {
    if (!results.counterfactual) return null;

    const data = Object.entries(results.counterfactual)
      .filter(([key]) => key.startsWith('effect_'))
      .map(([key, value]) => ({
        gene: key.replace('effect_', ''),
        effect: value
      }));

    return (
      <div className="result-section">
        <h4><GitBranch size={16} /> Counterfactual Analysis</h4>
        <div className="baseline-metric">
          <span>Baseline: {results.counterfactual.baseline?.toFixed(3)}</span>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="gene" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="effect" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderBackdoorResults = () => {
    if (!results.backdoor) return null;

    return (
      <div className="result-section">
        <h4><Activity size={16} /> Confounding Control</h4>
        <div className="causal-effect-card">
          <div className="ate-value">
            <span className="label">Average Treatment Effect:</span>
            <span className="value">{results.backdoor.ate?.toFixed(3)}</span>
          </div>
          <div className="confidence-interval">
            <span className="label">95% CI:</span>
            <span className="value">
              [{results.backdoor.confidence_interval?.[0]?.toFixed(3)}, 
               {results.backdoor.confidence_interval?.[1]?.toFixed(3)}]
            </span>
          </div>
          <div className="p-value">
            <span className="label">p-value:</span>
            <span className={`value ${results.backdoor.p_value < 0.05 ? 'significant' : ''}`}>
              {results.backdoor.p_value?.toFixed(3)}
            </span>
          </div>
          {results.backdoor.confounders_adjusted?.length > 0 && (
            <div className="confounders">
              <span className="label">Adjusted confounders:</span>
              <div className="confounder-list">
                {results.backdoor.confounders_adjusted.map((conf, idx) => (
                  <span key={idx} className="confounder-tag">{conf}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMediationResults = () => {
    if (!results.mediation) return null;

    const data = [
      { name: 'Total Effect', value: results.mediation.total_effect },
      { name: 'Direct Effect', value: results.mediation.direct_effect },
      { name: 'Indirect Effect', value: results.mediation.indirect_effect }
    ];

    return (
      <div className="result-section">
        <h4><TrendingUp size={16} /> Mediation Analysis</h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
        <div className="mediation-proportion">
          <span className="label">Mediation Proportion:</span>
          <span className="value">
            {(results.mediation.mediation_proportion * 100).toFixed(1)}%
          </span>
        </div>
        {results.mediation.mediator && (
          <div className="mediator-info">
            <span className="label">Mediator:</span>
            <span className="value">{results.mediation.mediator}</span>
          </div>
        )}
      </div>
    );
  };

  const renderNeuroSymbolicResults = () => {
    if (!results.neuro_symbolic) return null;

    const { neural_component, symbolic_component, combined_prediction } = results.neuro_symbolic;

    return (
      <div className="result-section">
        <h4><Activity size={16} /> Neuro-Symbolic Analysis</h4>
        
        <div className="neuro-symbolic-grid">
          <div className="neural-component">
            <h5>Neural Component</h5>
            <div className="component-card">
              <div className="metric-value">{neural_component.probability?.toFixed(3)}</div>
              <div className="metric-label">GNN Prediction</div>
              <div className="component-weight">Weight: {(neural_component.weight * 100).toFixed(0)}%</div>
            </div>
          </div>
          
          <div className="symbolic-component">
            <h5>Symbolic Component</h5>
            <div className="component-card">
              <div className="metric-value">{symbolic_component.rule_confidence?.toFixed(3)}</div>
              <div className="metric-label">Rule Confidence</div>
              <div className="component-weight">Weight: {(symbolic_component.weight * 100).toFixed(0)}%</div>
              <div className="pathway-status">
                Pathway: {symbolic_component.pathway_exists ? 'Exists' : 'Missing'}
              </div>
            </div>
          </div>
        </div>
        
        <div className="combined-prediction">
          <h5>Combined Prediction</h5>
          <div className="metric-card">
            <div className="metric-value">{combined_prediction.probability?.toFixed(3)}</div>
            <div className="metric-label">Neuro-Symbolic Result</div>
          </div>
        </div>
        
        {symbolic_component.related_pathways?.length > 0 && (
          <div className="pathways-section">
            <h5>Related Pathways</h5>
            <div className="pathway-list">
              {symbolic_component.related_pathways.map((pathway, idx) => (
                <div key={idx} className="pathway-item">
                  <span className="pathway-gene">{pathway.gene}</span>
                  <span className="pathway-confidence">{(pathway.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {symbolic_component.rule_violations?.length > 0 && (
          <div className="rules-section">
            <h5>Rule Violations</h5>
            <div className="rule-list">
              {symbolic_component.rule_violations.map((rule, idx) => (
                <div key={idx} className="rule-item">
                  <span className="rule-pair">{rule.drug} → {rule.disease}</span>
                  <span className="rule-confidence">{(rule.confidence * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderLLMExplanation = () => {
    if (!results.llm_explanation) return null;

    return (
      <div className="result-section llm-explanation">
        <h4><MessageSquare size={16} /> AI Analysis</h4>
        <div className="explanation-text">
          {results.llm_explanation}
        </div>
      </div>
    );
  };

  const renderNaturalLanguageResults = () => {
    if (!results.answer) return null;

    return (
      <div className="result-section">
        <h4><MessageSquare size={16} /> Analysis Explanation</h4>
        <div className="explanation-text">
          {results.answer}
        </div>
      </div>
    );
  };

  return (
    <div className="results-visualization">
      <div className="results-header">
        <CheckCircle size={20} className="success-icon" />
        <h3>Causal Analysis Results</h3>
      </div>

      <div className="results-content">
        {renderInterventionalResults()}
        {renderCounterfactualResults()}
        {renderBackdoorResults()}
        {renderMediationResults()}
        {renderNeuroSymbolicResults()}
        {renderNaturalLanguageResults()}
        {renderLLMExplanation()}
        
        <div className="graph-toggle">
          <button 
            className="toggle-button" 
            onClick={() => setShowGraph(!showGraph)}
            disabled={loadingGraph}
          >
            {showGraph ? 'Hide' : 'Show'} Knowledge Graph
          </button>
        </div>
        
        {showGraph && (
          loadingGraph ? (
            <div className="graph-placeholder">
              <p>Loading knowledge graph...</p>
            </div>
          ) : (
            <KnowledgeGraphVisualization graphData={graphData} />
          )
        )}
      </div>
    </div>
  );
};

export default ResultsVisualization;