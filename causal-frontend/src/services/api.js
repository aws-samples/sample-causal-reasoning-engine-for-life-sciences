import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const causalAPI = {
  // Backend status
  getStatus: async () => {
    const response = await api.get('/status');
    return response.data;
  },

  // Node data
  getNodeData: async () => {
    const response = await api.get('/nodes');
    return response.data;
  },

  // Causal analysis methods
  performCausalAnalysis: async (queryData) => {
    const response = await api.post('/causal/analyze', queryData);
    return response.data;
  },

  // Interventional prediction
  interventionalPrediction: async (treatmentIdx, outcomeIdx, interventionValue = 1.0) => {
    const response = await api.post('/causal/intervention', {
      treatment_idx: treatmentIdx,
      outcome_idx: outcomeIdx,
      intervention_value: interventionValue
    });
    return response.data;
  },

  // Counterfactual analysis
  counterfactualAnalysis: async (treatmentIdx, outcomeIdx, geneInterventions) => {
    const response = await api.post('/causal/counterfactual', {
      treatment_idx: treatmentIdx,
      outcome_idx: outcomeIdx,
      gene_interventions: geneInterventions
    });
    return response.data;
  },

  // Backdoor adjustment
  backdoorAdjustment: async (treatmentIdx, outcomeIdx) => {
    const response = await api.post('/causal/backdoor', {
      treatment_idx: treatmentIdx,
      outcome_idx: outcomeIdx
    });
    return response.data;
  },

  // Mediation analysis
  mediationAnalysis: async (treatmentIdx, mediatorIdx, outcomeIdx) => {
    const response = await api.post('/causal/mediation', {
      treatment_idx: treatmentIdx,
      mediator_idx: mediatorIdx,
      outcome_idx: outcomeIdx
    });
    return response.data;
  },

  // Neuro-symbolic analysis
  neuroSymbolicAnalysis: async (treatmentIdx, outcomeIdx) => {
    const response = await api.post('/neuro-symbolic/analyze', {
      treatment_idx: treatmentIdx,
      outcome_idx: outcomeIdx
    });
    return response.data;
  },

  // Knowledge graph subgraph
  getSubgraph: async (treatmentIdx, outcomeIdx) => {
    const response = await api.post('/graph/subgraph', {
      treatment_idx: treatmentIdx,
      outcome_idx: outcomeIdx
    });
    return response.data;
  },


};

export default api;