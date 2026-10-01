# Causal LLM Analysis Frontend

A React-based web interface for causal analysis of biomedical knowledge graphs, featuring interventional predictions, counterfactual analysis, and confounding control.

## Quick Start

1. **Start Backend API:**
```bash
.venv/bin/python causal_api.py
```

2. **Start Frontend (in new terminal):**
```bash
cd causal-frontend
npm install  # First time only
npm start
```

## Access Points

- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:5000
- **API Status:** http://localhost:5000/status

## Features

### Causal Analysis Types
- **Interventional Prediction:** P(Y|do(X)) - What happens if we intervene?
- **Counterfactual Analysis:** What if we changed gene expression?
- **Backdoor Adjustment:** Control for confounding variables
- **Mediation Analysis:** Direct vs indirect causal pathways

### Interactive Components
- **Query Interface:** Ask natural language causal questions
- **Analysis Panel:** Configure specific causal analyses
- **Results Visualization:** Charts and metrics for causal effects
- **Node Explorer:** Browse compounds, genes, and diseases

## Example Queries

Try these causal questions in the interface:

```
"What is the causal effect of compound-0 on disease-0?"
"What would happen if we intervened on gene expression?"
"How do confounders affect this drug-disease relationship?"
"What is the mediation pathway from drug to disease?"
"What if we knocked out gene-5 while giving compound-2?"
```

## Configuration

### Backend Configuration
- Host, port, and debug flags: `CAUSAL_API_HOST`, `CAUSAL_API_PORT`, `CAUSAL_API_DEBUG`
- Data directory: Update paths in `load_pennsieve()`
- Model parameters: Adjust in `CausalGNN` initialization

### Frontend Configuration
- API URL: Set `REACT_APP_API_URL` environment variable
- Styling: Modify `src/App.css`
- Components: Customize in `src/components/`

## Project Structure

```
CausalLLM/
├── causal-frontend/          # React frontend application
│   ├── src/
│   │   ├── components/       # UI components
│   │   ├── services/         # API integration
│   │   └── App.js           # Main application
│   ├── public/
│   └── package.json
├── causal_api.py            # Flask backend API
├── causal_models.py         # Knowledge graph and GNN model
├── causal_training.py       # Training loop
└── causal_inference.py      # Standalone inference server
```

## Causal Analysis Methods

### 1. Interventional Prediction
Estimates P(Y|do(X)) - the probability of outcome Y when we intervene to set treatment X.

### 2. Counterfactual Analysis
Answers "what if" questions by simulating alternative scenarios with gene interventions.

### 3. Backdoor Adjustment
Controls for confounding variables to estimate unbiased causal effects.

### 4. Mediation Analysis
Decomposes total causal effects into direct and indirect pathways through mediators.

## Dependencies

### Backend (Python)
- Flask, Flask-CORS
- PyTorch
- NumPy, Pandas
- NetworkX
- Scikit-learn
- Boto3 (for AWS Bedrock LLM)

### Frontend (Node.js)
- React 18+
- Axios (API calls)
- Recharts (visualizations)
- Lucide React (icons)

## Troubleshooting

### Backend Issues
- **Port 5000 in use:** Change port in `causal_api.py`
- **Missing data:** Check data directory paths
- **Model errors:** Verify PyTorch installation

### Frontend Issues
- **API connection failed:** Ensure backend is running on port 5000
- **Build errors:** Run `npm install` in causal-frontend directory
- **CORS errors:** Check Flask-CORS configuration

## Usage Examples

### 1. Basic Causal Query
1. Open http://localhost:3000
2. Type: "What is the causal effect of compound-0 on disease-0?"
3. View results in the visualization panel

### 2. Interventional Analysis
1. Select "Interventional" in the analysis panel
2. Choose treatment (compound) and outcome (disease)
3. Click "Run Causal Analysis"
4. Interpret P(Y|do(X)) probability

### 3. Counterfactual Scenario
1. Select "Counterfactual" analysis type
2. Add gene interventions with sliders
3. Compare baseline vs intervention effects

## Interpreting Results

- **ATE (Average Treatment Effect):** Mean causal effect size
- **Confidence Intervals:** Uncertainty bounds around estimates
- **P-values:** Statistical significance of causal effects
- **Mediation Proportion:** % of effect through indirect pathways

## Future Enhancements

- Real-time model training interface
- Advanced visualization options
- Export results to CSV/PDF
- Integration with external databases
- Multi-user collaboration features

## Notes

- The system uses synthetic data by default if real biomedical data isn't available
- Causal analysis requires careful interpretation of assumptions
- Results should be validated with domain expertise
- LLM explanations require AWS Bedrock access

---