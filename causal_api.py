import os
import sys
import json
import traceback
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import boto3

from causal_models import (
    load_pennsieve, 
    CausalGNN, 
    HeterogeneousKnowledgeGraph
)

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 5000

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'], supports_credentials=True)

knowledge_graph: Optional[HeterogeneousKnowledgeGraph] = None
model: Optional[CausalGNN] = None
analyzer = None
bedrock_client = None

def enhance_with_llm(analysis_result: Dict[str, Any], treatment_idx: int, outcome_idx: int) -> Dict[str, Any]:
    """Enhance causal analysis results with LLM explanation"""
    if not bedrock_client or not knowledge_graph:
        return analysis_result
    
    try:
        treatment_name = knowledge_graph.node_names["Compound"].get(treatment_idx, f"Compound-{treatment_idx}")
        outcome_name = knowledge_graph.node_names["Disease"].get(outcome_idx, f"Disease-{outcome_idx}")
        
        relevant_genes = []
        for protein_idx, disease_idx in knowledge_graph.protein_to_disease:
            if disease_idx == outcome_idx:
                gene_name = knowledge_graph.node_names["Gene"].get(protein_idx, f"Gene-{protein_idx}")
                relevant_genes.append(gene_name)
        
        graph_context = f"Treatment: {treatment_name}, Outcome: {outcome_name}, Related genes: {', '.join(relevant_genes[:5])}"
        
        prompt = f"""You are a causal inference expert analyzing biomedical data.

Causal Analysis Results: {json.dumps(analysis_result, indent=2)}
Graph Context: {graph_context}

Provide a clear explanation focusing on:
1. What the causal effect means
2. Clinical implications
3. Biological mechanisms

Keep it concise and scientifically accurate."""
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        })
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        llm_explanation = response_body['content'][0]['text']
        
        analysis_result['llm_explanation'] = llm_explanation
        return analysis_result
        
    except Exception as e:
        print(f"LLM enhancement error: {e}")
        return analysis_result

def initialize_models():
    """Initialize the causal analysis models"""
    global knowledge_graph, model, analyzer, bedrock_client
    
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        print("Loading causal knowledge graph...")
        data_directory = os.path.join(os.path.dirname(__file__), "data")
        knowledge_graph = load_pennsieve(data_directory)
        
        print(f"Loaded KG: {knowledge_graph.num_drugs} compounds, {knowledge_graph.num_proteins} genes, {knowledge_graph.num_diseases} diseases")
        
        print("Initializing causal model...")
        model = CausalGNN(
            drug_input_dim=knowledge_graph.drug_dim,
            protein_input_dim=knowledge_graph.protein_dim, 
            disease_input_dim=knowledge_graph.disease_dim,
            hidden_dim=128
        )
        
        

        
        print("Models initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        traceback.print_exc()
        return False



@app.route('/status', methods=['GET'])
def get_status():
    """Get API status"""
    if model is None:
        return jsonify({"status": "disconnected", "message": "Models not initialized"})
    return jsonify({"status": "connected", "message": "Causal analysis ready"})

@app.route('/nodes', methods=['GET'])
def get_nodes():
    """Get node data for the frontend"""
    if knowledge_graph is None:
        return jsonify({"error": "Knowledge graph not loaded"}), 500
    
    try:
        compounds = [{"name": name, "id": idx} 
                    for idx, name in knowledge_graph.node_names.get("Compound", {}).items()]
        genes = [{"name": name, "id": idx} 
                for idx, name in knowledge_graph.node_names.get("Gene", {}).items()]
        diseases = [{"name": name, "id": idx} 
                   for idx, name in knowledge_graph.node_names.get("Disease", {}).items()]
        
        return jsonify({
            "compounds": compounds,
            "genes": genes, 
            "diseases": diseases,
            "stats": {
                "num_compounds": knowledge_graph.num_drugs,
                "num_genes": knowledge_graph.num_proteins,
                "num_diseases": knowledge_graph.num_diseases,
                "num_protein_disease_edges": len(knowledge_graph.protein_to_disease)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/causal/analyze', methods=['POST'])
def analyze_causal():
    """Main causal analysis endpoint"""
    if knowledge_graph is None or model is None:
        return jsonify({"error": "Models not initialized"}), 500
    
    try:
        data = request.get_json()
        analysis_type = data.get('analysis_type')
        
        if analysis_type == 'intervention':
            return handle_intervention(data)
        elif analysis_type == 'counterfactual':
            return handle_counterfactual(data)
        elif analysis_type == 'backdoor':
            return handle_backdoor(data)
        elif analysis_type == 'mediation':
            return handle_mediation(data)
        else:
            return jsonify({"error": f"Unknown analysis type: {analysis_type}"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_intervention(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return an interventional estimate.

    NOTE: returns fixed sample values. Wire in a causal analyzer for real estimates.
    """
    treatment_index = data.get('treatment_idx', 0)
    outcome_index = data.get('outcome_idx', 0)
    intervention_value = data.get('intervention_value', 1.0)
    
    probability, uncertainty = 0.5, 0.1
    
    result = {
        "interventional": {
            "probability": float(probability),
            "uncertainty": float(uncertainty),
            "treatment": knowledge_graph.node_names["Compound"].get(treatment_index, f"Compound-{treatment_index}"),
            "outcome": knowledge_graph.node_names["Disease"].get(outcome_index, f"Disease-{outcome_index}")
        }
    }
    
    enhanced_result = enhance_with_llm(result, treatment_index, outcome_index)
    return jsonify(enhanced_result)

def handle_counterfactual(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a counterfactual estimate.

    NOTE: returns fixed sample values. Wire in a causal analyzer for real estimates.
    """
    treatment_index = data.get('treatment_idx', 0)
    outcome_index = data.get('outcome_idx', 0)
    gene_interventions = data.get('gene_interventions', {})
    
    gene_interventions = {int(key): float(value) for key, value in gene_interventions.items()}
    
    result = {"counterfactual": {"counterfactual_probability": 0.6, "confidence": 0.8}}
    
    enhanced_result = enhance_with_llm(result, treatment_index, outcome_index)
    return jsonify(enhanced_result)

def handle_backdoor(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a backdoor-adjusted treatment effect.

    NOTE: returns fixed sample values. Wire in a causal analyzer for real estimates.
    """
    treatment_index = data.get('treatment_idx', 0)
    outcome_index = data.get('outcome_idx', 0)
    
    class BackdoorResult:
        def __init__(self):
            self.ate = 0.3
            self.confidence_interval = [0.1, 0.5]
            self.p_value = 0.05
            self.confounders_adjusted = ["gene_1", "gene_2"]
    backdoor_result = BackdoorResult()
    
    result = {
        "backdoor": {
            "average_treatment_effect": backdoor_result.ate,
            "confidence_interval": backdoor_result.confidence_interval,
            "p_value": backdoor_result.p_value,
            "confounders_adjusted": backdoor_result.confounders_adjusted
        }
    }
    
    enhanced_result = enhance_with_llm(result, treatment_index, outcome_index)
    return jsonify(enhanced_result)

def handle_mediation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a mediation decomposition.

    NOTE: returns fixed sample values. Wire in a causal analyzer for real estimates.
    """
    treatment_index = data.get('treatment_idx', 0)
    mediator_index = data.get('mediator_idx', 0)
    outcome_index = data.get('outcome_idx', 0)
    
    result = {"mediation": {"direct_effect": 0.2, "indirect_effect": 0.1, "total_effect": 0.3}}
    
    enhanced_result = enhance_with_llm(result, treatment_index, outcome_index)
    return jsonify(enhanced_result)



@app.route('/causal/intervention', methods=['POST'])
def intervention_endpoint():
    """Dedicated interventional prediction endpoint"""
    data = request.get_json()
    return handle_intervention(data)

@app.route('/causal/counterfactual', methods=['POST'])
def counterfactual_endpoint():
    """Dedicated counterfactual analysis endpoint"""
    data = request.get_json()
    return handle_counterfactual(data)

@app.route('/causal/backdoor', methods=['POST'])
def backdoor_endpoint():
    """Dedicated backdoor adjustment endpoint"""
    data = request.get_json()
    return handle_backdoor(data)

@app.route('/causal/mediation', methods=['POST'])
def mediation_endpoint():
    """Dedicated mediation analysis endpoint"""
    data = request.get_json()
    return handle_mediation(data)

@app.route('/graph/subgraph', methods=['POST'])
def get_subgraph():
    """Get knowledge graph subgraph for visualization"""
    try:
        data = request.get_json()
        treatment_idx = data.get('treatment_idx', 0)
        outcome_idx = data.get('outcome_idx', 0)
        max_nodes = data.get('max_nodes', 20)
        
        nodes = []
        edges = []
        
        treatment_name = knowledge_graph.node_names["Compound"].get(treatment_idx, f"Compound-{treatment_idx}")
        nodes.append({'id': f'drug_{treatment_idx}', 'name': treatment_name, 'type': 'drug', 'group': 1})
        
        outcome_name = knowledge_graph.node_names["Disease"].get(outcome_idx, f"Disease-{outcome_idx}")
        nodes.append({'id': f'disease_{outcome_idx}', 'name': outcome_name, 'type': 'disease', 'group': 3})
        
        connecting_proteins = set()
        for protein_idx, disease_idx in knowledge_graph.protein_to_disease:
            if disease_idx == outcome_idx:
                connecting_proteins.add(protein_idx)
        
        # Add protein nodes and edges (limit to max_nodes)
        for protein_idx in list(connecting_proteins)[:max_nodes-2]:
            protein_name = knowledge_graph.node_names["Gene"].get(protein_idx, f"Gene-{protein_idx}")
            nodes.append({'id': f'protein_{protein_idx}', 'name': protein_name, 'type': 'protein', 'group': 2})
            
            edges.append({
                'source': f'protein_{protein_idx}',
                'target': f'disease_{outcome_idx}',
                'type': 'affects'
            })
            
            for drug_idx, prot_idx in knowledge_graph.drug_to_protein_inhibition:
                if drug_idx == treatment_idx and prot_idx == protein_idx:
                    edges.append({
                        'source': f'drug_{treatment_idx}',
                        'target': f'protein_{protein_idx}',
                        'type': 'inhibits'
                    })
            
            for drug_idx, prot_idx in knowledge_graph.drug_to_protein_activation:
                if drug_idx == treatment_idx and prot_idx == protein_idx:
                    edges.append({
                        'source': f'drug_{treatment_idx}',
                        'target': f'protein_{protein_idx}',
                        'type': 'activates'
                    })
        
        return jsonify({'nodes': nodes, 'edges': edges})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting Causal LLM API...")

    # Bind to loopback and disable the debugger by default. Override via
    # environment for container or reverse-proxy deployments, where the
    # listener must be reachable from outside the container's network namespace.
    host = os.environ.get('CAUSAL_API_HOST', DEFAULT_HOST)
    port = int(os.environ.get('CAUSAL_API_PORT', DEFAULT_PORT))
    debug = os.environ.get('CAUSAL_API_DEBUG', '').lower() in ('1', 'true', 'yes')

    if initialize_models():
        print("Models loaded successfully")
        print(f"Starting Flask server on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
    else:
        print("Failed to initialize models")
        sys.exit(1)