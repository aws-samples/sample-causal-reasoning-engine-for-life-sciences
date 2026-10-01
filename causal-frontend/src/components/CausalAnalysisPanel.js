import React, { useState } from 'react';
import { Activity, Zap, GitBranch, TrendingUp } from 'lucide-react';

const CausalAnalysisPanel = ({ onQuery, loading, nodeData }) => {
  const [analysisType, setAnalysisType] = useState('intervention');
  const [treatmentIdx, setTreatmentIdx] = useState(0);
  const [outcomeIdx, setOutcomeIdx] = useState(0);
  const [mediatorIdx, setMediatorIdx] = useState(0);
  const [geneInterventions, setGeneInterventions] = useState({});

  const analysisTypes = [
    { id: 'intervention', name: 'Interventional', icon: Zap, desc: 'P(Y|do(X))' },
    { id: 'counterfactual', name: 'Counterfactual', icon: GitBranch, desc: 'What if?' },
    { id: 'backdoor', name: 'Confounding Control', icon: Activity, desc: 'Adjust confounders' },
    { id: 'mediation', name: 'Mediation', icon: TrendingUp, desc: 'Direct vs indirect' },
    { id: 'neuro-symbolic', name: 'Neuro-Symbolic', icon: Activity, desc: 'Neural + Symbolic' }
  ];

  const handleAnalysis = () => {
    const queryData = {
      analysis_type: analysisType,
      treatment_idx: treatmentIdx,
      outcome_idx: outcomeIdx,
      mediator_idx: mediatorIdx,
      gene_interventions: geneInterventions
    };
    onQuery(queryData);
  };

  const addGeneIntervention = () => {
    // Find the first available gene index not already in interventions
    const usedIndices = new Set(Object.keys(geneInterventions).map(k => parseInt(k)));
    const maxGenes = nodeData?.genes?.length || 100;
    let geneIdx = 0;
    while (usedIndices.has(geneIdx) && geneIdx < maxGenes) {
      geneIdx++;
    }
    if (geneIdx < maxGenes) {
      setGeneInterventions({
        ...geneInterventions,
        [geneIdx]: 0.5
      });
    }
  };

  const updateGeneIntervention = (geneIdx, value) => {
    setGeneInterventions({
      ...geneInterventions,
      [geneIdx]: parseFloat(value)
    });
  };

  const removeGeneIntervention = (geneIdx) => {
    const newInterventions = { ...geneInterventions };
    delete newInterventions[geneIdx];
    setGeneInterventions(newInterventions);
  };

  return (
    <div className="causal-analysis-panel">
      <h3>Causal Analysis</h3>
      
      {/* Analysis Type Selection */}
      <div className="analysis-types">
        {analysisTypes.map(type => {
          const Icon = type.icon;
          return (
            <div
              key={type.id}
              className={`analysis-type ${analysisType === type.id ? 'active' : ''}`}
              onClick={() => setAnalysisType(type.id)}
            >
              <Icon size={16} />
              <div>
                <div className="type-name">{type.name}</div>
                <div className="type-desc">{type.desc}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Parameters */}
      <div className="parameters">
        <div className="param-group">
          <label>Treatment (Compound)</label>
          <select 
            value={treatmentIdx} 
            onChange={(e) => setTreatmentIdx(parseInt(e.target.value))}
          >
            {nodeData?.compounds?.map((compound, idx) => (
              <option key={idx} value={idx}>
                {compound.name || `Compound-${idx}`}
              </option>
            )) || Array.from({length: 50}, (_, i) => (
              <option key={i} value={i}>Compound-{i}</option>
            ))}
          </select>
        </div>

        <div className="param-group">
          <label>Outcome (Disease)</label>
          <select 
            value={outcomeIdx} 
            onChange={(e) => setOutcomeIdx(parseInt(e.target.value))}
          >
            {nodeData?.diseases?.map((disease, idx) => (
              <option key={idx} value={idx}>
                {disease.name || `Disease-${idx}`}
              </option>
            )) || Array.from({length: 50}, (_, i) => (
              <option key={i} value={i}>Disease-{i}</option>
            ))}
          </select>
        </div>

        {analysisType === 'mediation' && (
          <div className="param-group">
            <label>Mediator (Gene)</label>
            <select 
              value={mediatorIdx} 
              onChange={(e) => setMediatorIdx(parseInt(e.target.value))}
            >
              {nodeData?.genes?.map((gene, idx) => (
                <option key={idx} value={idx}>
                  {gene.name || `Gene-${idx}`}
                </option>
              )) || Array.from({length: 100}, (_, i) => (
                <option key={i} value={i}>Gene-{i}</option>
              ))}
            </select>
          </div>
        )}

        {analysisType === 'counterfactual' && (
          <div className="param-group">
            <label>Gene Interventions</label>
            <div className="gene-interventions">
              {Object.entries(geneInterventions).map(([geneIdx, value]) => {
                const geneName = nodeData?.genes?.[parseInt(geneIdx)]?.name || `Gene-${geneIdx}`;
                return (
                  <div key={geneIdx} className="gene-intervention">
                    <select 
                      value={geneIdx}
                      onChange={(e) => {
                        const newGeneIdx = e.target.value;
                        const newInterventions = { ...geneInterventions };
                        delete newInterventions[geneIdx];
                        newInterventions[newGeneIdx] = value;
                        setGeneInterventions(newInterventions);
                      }}
                    >
                      {nodeData?.genes?.map((gene, idx) => (
                        <option key={idx} value={idx}>
                          {gene.name || `Gene-${idx}`}
                        </option>
                      )) || Array.from({length: 100}, (_, i) => (
                        <option key={i} value={i}>Gene-{i}</option>
                      ))}
                    </select>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={value}
                      onChange={(e) => updateGeneIntervention(geneIdx, e.target.value)}
                    />
                    <span>{value}</span>
                    <button onClick={() => removeGeneIntervention(geneIdx)}>×</button>
                  </div>
                );
              })}
              <button className="add-intervention" onClick={addGeneIntervention}>
                + Add Gene Intervention
              </button>
            </div>
          </div>
        )}
      </div>

      <button 
        className="analyze-btn"
        onClick={handleAnalysis}
        disabled={loading}
      >
        {loading ? 'Analyzing...' : 'Run Causal Analysis'}
      </button>
    </div>
  );
};

export default CausalAnalysisPanel;