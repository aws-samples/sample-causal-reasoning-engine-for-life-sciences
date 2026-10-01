import sys
sys.path.append('.')
from causal_models import load_pennsieve, CausalGNN
import json
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import os
import boto3

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class CausalInferenceServer:
    def __init__(self, pennsieve_path: str, model_path: str = None, port: int = 65201):
        self.port = port
        self.kg = None
        self.model = None
        self.qa_system = None
        self._initialize_system(pennsieve_path, model_path)
    
    def _load_trained_model(self, model_path: str):
        """Load a trained model from file"""
        if os.path.exists(model_path):
            print(f'Loading trained model from {model_path}')
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            print(f'Checkpoint keys: {list(checkpoint.keys())}')
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f'Model config: {config}')
                hidden_dimension = config.get('hidden_dim', config.get('h', 64))
                print(f'Using hidden dimension: {hidden_dimension}')
                self.model = CausalGNN(config.get('drug_input_dim', config.get('din', 96)), 
                                     config.get('protein_input_dim', config.get('pin', 96)), 
                                     config.get('disease_input_dim', config.get('xin', 96)), 
                                     hidden_dimension)
            else:
                print('No model_config found, using default h=64')
                self.model = CausalGNN(self.kg.drug_dim, self.kg.protein_dim, self.kg.disease_dim, hidden_dim=64)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            if 'kg_data' in checkpoint:
                kg_data = checkpoint['kg_data']
                self.kg.node_names = kg_data['node_names']
                print('Loaded trained model with saved knowledge graph data')
            
            return True
        return False
    
    def _initialize_system(self, pennsieve_path: str, model_path: str = None):
        """Initialize the causal reasoning system"""
        print('Initializing causal system...')
        self.kg = load_pennsieve(pennsieve_path, dimension=96)
        print('Loaded Pennsieve dataset with real entity names')
        print(f'Debug: Sample compound names: {list(self.kg.node_names["Compound"].values())[:5]}')
        print(f'Debug: Sample gene names: {list(self.kg.node_names["Gene"].values())[:5]}')
        print(f'Debug: Sample disease names: {list(self.kg.node_names["Disease"].values())[:5]}')
        
        if model_path and self._load_trained_model(model_path):
            print('Using trained model for inference')
        else:
            print('No trained model found, training new model...')
            from causal_training import train_model
            self.model, _ = train_model(self.kg, epochs=10, save_path=model_path or "causal_model.pth")
            print('Model training completed')
        
        print(f'System ready: {self.kg.num_drugs} compounds, {self.kg.num_proteins} genes, {self.kg.num_diseases} diseases')
        
        self.qa_system = None
    
    def get_status(self):
        """Get system status"""
        return {'status': 'connected', 'message': 'Backend ready'}
    
    def get_nodes(self):
        """Get available nodes for frontend"""
        print(f'Debug: First 3 compound names: {[self.kg.node_names["Compound"].get(i, f"Missing-{i}") for i in range(min(3, self.kg.num_drugs))]}')
        print(f'Debug: First 3 gene names: {[self.kg.node_names["Gene"].get(i, f"Missing-{i}") for i in range(min(3, self.kg.num_proteins))]}')
        print(f'Debug: First 3 disease names: {[self.kg.node_names["Disease"].get(i, f"Missing-{i}") for i in range(min(3, self.kg.num_diseases))]}')
        
        return {
            'compounds': [{'name': self.kg.node_names['Compound'].get(i, f'Unknown-Compound-{i}'), 'id': i} for i in range(self.kg.num_drugs)],
            'genes': [{'name': self.kg.node_names['Gene'].get(i, f'Unknown-Gene-{i}'), 'id': i} for i in range(self.kg.num_proteins)],
            'diseases': [{'name': self.kg.node_names['Disease'].get(i, f'Unknown-Disease-{i}'), 'id': i} for i in range(self.kg.num_diseases)]
        }
    
    def get_subgraph(self, treatment_idx, outcome_idx, max_nodes=20):
        """Get relevant subgraph for visualization"""
        nodes = []
        edges = []
        
        treatment_name = self.kg.node_names["Compound"].get(treatment_idx, f"Compound-{treatment_idx}")
        nodes.append({'id': f'drug_{treatment_idx}', 'name': treatment_name, 'type': 'drug', 'group': 1})
        
        outcome_name = self.kg.node_names["Disease"].get(outcome_idx, f"Disease-{outcome_idx}")
        nodes.append({'id': f'disease_{outcome_idx}', 'name': outcome_name, 'type': 'disease', 'group': 3})
        
        connecting_proteins = set()
        for protein_idx, disease_idx in self.kg.protein_to_disease:
            if disease_idx == outcome_idx:
                connecting_proteins.add(protein_idx)
        
        # Add protein nodes and edges (limit to max_nodes)
        protein_count = 0
        for protein_idx in list(connecting_proteins)[:max_nodes-2]:
            protein_name = self.kg.node_names["Gene"].get(protein_idx, f"Gene-{protein_idx}")
            nodes.append({'id': f'protein_{protein_idx}', 'name': protein_name, 'type': 'protein', 'group': 2})
            
            # Add protein-disease edge
            edges.append({
                'source': f'protein_{protein_idx}',
                'target': f'disease_{outcome_idx}',
                'type': 'affects'
            })
            
            # Check for drug-protein interactions
            for drug_idx, prot_idx in self.kg.drug_to_protein_inhibition:
                if drug_idx == treatment_idx and prot_idx == protein_idx:
                    edges.append({
                        'source': f'drug_{treatment_idx}',
                        'target': f'protein_{protein_idx}',
                        'type': 'inhibits'
                    })
            
            for drug_idx, prot_idx in self.kg.drug_to_protein_activation:
                if drug_idx == treatment_idx and prot_idx == protein_idx:
                    edges.append({
                        'source': f'drug_{treatment_idx}',
                        'target': f'protein_{protein_idx}',
                        'type': 'activates'
                    })
            
            protein_count += 1
        
        return {'nodes': nodes, 'edges': edges}
    
    def _enhance_with_llm(self, result, treatment_idx, outcome_idx):
        """Enhance results with LLM explanation"""
        try:
            bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
            
            treatment_name = self.kg.node_names["Compound"].get(treatment_idx, f"Compound-{treatment_idx}")
            outcome_name = self.kg.node_names["Disease"].get(outcome_idx, f"Disease-{outcome_idx}")
            
            relevant_genes = []
            for protein_idx, disease_idx in self.kg.protein_to_disease:
                if disease_idx == outcome_idx:
                    gene_name = self.kg.node_names["Gene"].get(protein_idx, f"Gene-{protein_idx}")
                    relevant_genes.append(gene_name)
            
            graph_context = f"Treatment: {treatment_name}, Outcome: {outcome_name}, Related genes: {', '.join(relevant_genes[:5])}"
            
            prompt = f"""You are a causal inference expert analyzing biomedical data.

Causal Analysis Results: {json.dumps(result, indent=2)}
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
            
            response = bedrock.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            result['llm_explanation'] = response_body['content'][0]['text']
            
        except Exception as e:
            print(f"LLM enhancement error: {e}")
        
        return result
    
    def perform_neuro_symbolic_analysis(self, data):
        """Perform neuro-symbolic analysis showing neural + symbolic components"""
        drug_features = torch.tensor(self.kg.drug_features, dtype=torch.float32)
        protein_features = torch.tensor(self.kg.protein_features, dtype=torch.float32)
        disease_features = torch.tensor(self.kg.disease_features, dtype=torch.float32)
        protein_disease_edges = torch.tensor(self.kg.protein_to_disease, dtype=torch.long).t() if self.kg.protein_to_disease else torch.empty(2, 0, dtype=torch.long)
        protein_protein_edges = torch.tensor(self.kg.protein_to_protein, dtype=torch.long).t() if hasattr(self.kg, 'protein_to_protein') and self.kg.protein_to_protein else torch.empty(2, 0, dtype=torch.long)
        
        treatment_index = min(data.get('treatment_idx', 0), self.kg.num_drugs - 1)
        outcome_index = min(data.get('outcome_idx', 0), self.kg.num_diseases - 1)
        
        with torch.no_grad():
            drug_hidden, protein_hidden, disease_hidden = self.model(drug_features, protein_features, disease_features, protein_disease_edges, protein_protein_edges)
            
            # Neural component: Pure GNN prediction
            neural_logits, neural_uncertainty = self.model.predict(drug_hidden[treatment_index:treatment_index+1], disease_hidden[outcome_index:outcome_index+1])
            neural_probability = torch.sigmoid(neural_logits).item()
            
            # Symbolic component: Rule-based analysis
            pathway_exists = any((protein_idx, outcome_index) in self.kg.protein_to_disease for protein_idx in range(self.kg.num_proteins))
            rule_confidence = 0.8 if pathway_exists else 0.2
            
            # Rule violations check
            rule_violations = []
            for drug_idx in range(min(3, self.kg.num_drugs)):
                for disease_idx in range(min(3, self.kg.num_diseases)):
                    pathway_exists = any((protein_idx, disease_idx) in self.kg.protein_to_disease for protein_idx in range(self.kg.num_proteins))
                    if pathway_exists:
                        drug_name = self.kg.node_names["Compound"].get(drug_idx, f"Compound-{drug_idx}")
                        disease_name = self.kg.node_names["Disease"].get(disease_idx, f"Disease-{disease_idx}")
                        rule_violations.append({
                            'drug': drug_name,
                            'disease': disease_name,
                            'rule': 'pathway_exists',
                            'confidence': 0.8
                        })
            
            # Pathway information
            related_pathways = []
            for protein_idx, disease_idx in self.kg.protein_to_disease:
                if disease_idx == outcome_index:
                    gene_name = self.kg.node_names["Gene"].get(protein_idx, f"Gene-{protein_idx}")
                    related_pathways.append({
                        'gene': gene_name,
                        'confidence': 0.7 + 0.3 * torch.rand(1).item()
                    })
            
            # Combined neuro-symbolic prediction
            symbolic_weight = 0.3
            neural_weight = 0.7
            combined_probability = neural_weight * neural_probability + symbolic_weight * rule_confidence
            
            result = {
                'neuro_symbolic': {
                    'neural_component': {
                        'probability': neural_probability,
                        'uncertainty': neural_uncertainty.item(),
                        'weight': neural_weight
                    },
                    'symbolic_component': {
                        'rule_confidence': rule_confidence,
                        'pathway_exists': pathway_exists,
                        'weight': symbolic_weight,
                        'rule_violations': rule_violations[:5],
                        'related_pathways': related_pathways[:5]
                    },
                    'combined_prediction': {
                        'probability': combined_probability,
                        'integration_method': 'weighted_average'
                    },
                    'treatment': self.kg.node_names["Compound"].get(treatment_index, f"Compound-{treatment_index}"),
                    'outcome': self.kg.node_names["Disease"].get(outcome_index, f"Disease-{outcome_index}")
                }
            }
            
            return self._enhance_with_llm(result, treatment_index, outcome_index)
    
    def perform_causal_analysis(self, data):
        """Perform causal analysis based on request data"""
        drug_features = torch.tensor(self.kg.drug_features, dtype=torch.float32)
        protein_features = torch.tensor(self.kg.protein_features, dtype=torch.float32)
        disease_features = torch.tensor(self.kg.disease_features, dtype=torch.float32)
        protein_disease_edges = torch.tensor(self.kg.protein_to_disease, dtype=torch.long).t() if self.kg.protein_to_disease else torch.empty(2, 0, dtype=torch.long)
        protein_protein_edges = torch.tensor(self.kg.protein_to_protein, dtype=torch.long).t() if hasattr(self.kg, 'protein_to_protein') and self.kg.protein_to_protein else torch.empty(2, 0, dtype=torch.long)
        
        with torch.no_grad():
            drug_hidden, protein_hidden, disease_hidden = self.model(drug_features, protein_features, disease_features, protein_disease_edges, protein_protein_edges)
            treatment_index = min(data.get('treatment_idx', 0), self.kg.num_drugs - 1)
            outcome_index = min(data.get('outcome_idx', 0), self.kg.num_diseases - 1)
            analysis_type = data.get('analysis_type', 'intervention')
            
            # Modify embeddings based on analysis type to incorporate gene effects
            modified_drug_hidden = drug_hidden[treatment_index:treatment_index+1].clone()
            modified_disease_hidden = disease_hidden[outcome_index:outcome_index+1].clone()
            
            if analysis_type == 'mediation':
                mediator_idx = data.get('mediator_idx', 0)
                mediator_features = protein_hidden[mediator_idx:mediator_idx+1]
                # Incorporate mediator influence into disease embedding
                modified_disease_hidden = modified_disease_hidden + 0.3 * mediator_features
                
            elif analysis_type == 'counterfactual':
                gene_interventions = data.get('gene_interventions', {})
                # Apply gene interventions by modifying disease embedding
                for gene_idx_str, intervention_value in gene_interventions.items():
                    gene_idx = int(gene_idx_str)
                    if gene_idx < protein_hidden.shape[0]:
                        # Scale the influence based on intervention value
                        intervention_effect = (float(intervention_value) - 0.5) * 0.5  # Center around 0.5
                        modified_disease_hidden = modified_disease_hidden + intervention_effect * protein_hidden[gene_idx:gene_idx+1]
            
            # Get related proteins for this drug-disease pair
            related_proteins = [p_idx for p_idx, d_idx in self.kg.protein_to_disease if d_idx == outcome_index]
            if related_proteins:
                # Use the first related protein or the selected mediator for mediation analysis
                if analysis_type == 'mediation':
                    protein_idx = data.get('mediator_idx', related_proteins[0])
                else:
                    protein_idx = related_proteins[0]
                protein_features = protein_hidden[protein_idx:protein_idx+1]
                logits, uncertainty = self.model.predict(modified_drug_hidden, modified_disease_hidden, protein_features)
            else:
                # Fallback to no protein features if none are related
                logits, uncertainty = self.model.predict(modified_drug_hidden, modified_disease_hidden)
            probability = torch.sigmoid(logits).item()
            uncertainty_value = uncertainty.item()
            
            analysis_type = data.get('analysis_type', 'intervention')
            
            if analysis_type == 'intervention':
                result = {'interventional': {'probability': probability, 'uncertainty': uncertainty_value}}
            elif analysis_type == 'backdoor':
                result = {'backdoor': {'ate': probability-0.5, 'confidence_interval': [probability-uncertainty_value, probability+uncertainty_value], 'p_value': 0.05, 'confounders_adjusted': []}}
            elif analysis_type == 'counterfactual':
                gene_interventions = data.get('gene_interventions', {})
                counterfactual_effects = {'baseline': probability}
                
                if gene_interventions:
                    # Simplified approach: directly calculate effect based on intervention value and gene influence
                    for gene_idx_str, intervention_value in gene_interventions.items():
                        gene_idx = int(gene_idx_str)
                        gene_name = self.kg.node_names["Gene"].get(gene_idx, f"Gene-{gene_idx}")
                        
                        # Calculate effect based on intervention value and baseline probability
                        # intervention_value of 0.5 = no effect, >0.5 = positive effect, <0.5 = negative effect
                        intervention_strength = (float(intervention_value) - 0.5) * 2.0  # Scale to -1 to +1
                        effect = intervention_strength * probability * 0.3  # 30% max effect
                        
                        print(f"Debug: Gene {gene_name} intervention {intervention_value}, effect: {effect:.6f}")
                        counterfactual_effects[f'effect_{gene_name}'] = effect
                else:
                    # Fallback to first related gene if no interventions specified
                    for protein_idx, disease_idx in self.kg.protein_to_disease:
                        if disease_idx == outcome_index:
                            gene_name = self.kg.node_names["Gene"].get(protein_idx, f"Gene-{protein_idx}")
                            counterfactual_effects[f'effect_{gene_name}'] = 0.0  # No intervention
                            break
                
                result = {'counterfactual': counterfactual_effects}
            elif analysis_type == 'mediation':
                mediator_idx = data.get('mediator_idx', 0)
                mediator_name = self.kg.node_names["Gene"].get(mediator_idx, f"Gene-{mediator_idx}")
                
                # Calculate mediation effects using the selected gene
                # Direct effect: treatment -> outcome (without mediator)
                direct_logits, _ = self.model.predict(modified_drug_hidden, modified_disease_hidden)
                direct_probability = torch.sigmoid(direct_logits).item()
                
                # Indirect effect: treatment -> mediator -> outcome
                # Modify disease embedding based on mediator gene influence
                mediator_influenced_disease = modified_disease_hidden.clone()
                if mediator_idx < protein_hidden.shape[0]:
                    mediator_influence = 0.4 * protein_hidden[mediator_idx:mediator_idx+1]
                    mediator_influenced_disease = mediator_influenced_disease + mediator_influence
                
                if related_proteins:
                    mediator_features = protein_hidden[mediator_idx:mediator_idx+1]
                    indirect_logits, _ = self.model.predict(modified_drug_hidden, mediator_influenced_disease, mediator_features)
                else:
                    indirect_logits, _ = self.model.predict(modified_drug_hidden, mediator_influenced_disease)
                
                indirect_probability = torch.sigmoid(indirect_logits).item()
                
                # Calculate mediation components
                total_effect = probability  # Original baseline
                direct_effect = direct_probability - probability
                indirect_effect = indirect_probability - direct_probability
                mediation_proportion = abs(indirect_effect) / (abs(direct_effect) + abs(indirect_effect) + 1e-8)
                
                print(f"Debug: Mediation - Total: {total_effect:.4f}, Direct: {direct_effect:.4f}, Indirect: {indirect_effect:.4f}")
                
                result = {
                    'mediation': {
                        'total_effect': total_effect, 
                        'direct_effect': direct_effect, 
                        'indirect_effect': indirect_effect, 
                        'mediation_proportion': mediation_proportion,
                        'mediator': mediator_name
                    }
                }
            else:
                result = {'answer': f'Analysis complete: probability = {probability:.3f}'}
            
            return self._enhance_with_llm(result, treatment_index, outcome_index)
    


class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, inference_server, *args, **kwargs):
        self.inference_server = inference_server
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:3000')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:3000')
        self.end_headers()
        
        path = urlparse(self.path).path
        if path == '/status':
            response = self.inference_server.get_status()
        elif path == '/nodes':
            response = self.inference_server.get_nodes()
        else:
            response = {'error': 'Not found'}
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', 'http://localhost:3000')
        self.end_headers()
        
        path = urlparse(self.path).path
        print(f"POST request to path: {path}")
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            print(f"Request data: {data}")
            
            if path == '/causal/analyze':
                response = self.inference_server.perform_causal_analysis(data)
            elif path == '/neuro-symbolic/analyze':
                response = self.inference_server.perform_neuro_symbolic_analysis(data)
            elif path == '/graph/subgraph':
                treatment_idx = data.get('treatment_idx', 0)
                outcome_idx = data.get('outcome_idx', 0)
                response = self.inference_server.get_subgraph(treatment_idx, outcome_idx)
            else:
                response = self.inference_server.perform_causal_analysis(data)
        except Exception as e:
            print(f"Error in POST handler: {e}")
            import traceback
            traceback.print_exc()
            response = {'answer': f'Causal analysis error: {str(e)}'}
        
        self.wfile.write(json.dumps(response).encode())

def create_handler(inference_server):
    """Factory function to create request handler with inference server"""
    def handler(*args, **kwargs):
        return RequestHandler(inference_server, *args, **kwargs)
    return handler

def main():
    pennsieve_path = os.environ.get("PENNSIEVE_DATA_DIR", DEFAULT_DATA_DIR)
    model_path = "trained_causal_model.pth"  # Path to trained model
    port = 65201
    
    # Initialize inference server
    inference_server = CausalInferenceServer(pennsieve_path, model_path, port)
    
    # Create HTTP server
    handler = create_handler(inference_server)
    httpd = HTTPServer(('', port), handler)
    
    print(f'Causal inference server starting on port {port}')
    httpd.serve_forever()

if __name__ == "__main__":
    main()