import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
import os

from causal_models import CausalGNN, HeterogeneousKnowledgeGraph, TrainingResult, load_pennsieve

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class NeuroSymbolicLoss(nn.Module):
    def __init__(self, rule_weight=0.3, consistency_weight=0.2):
        super().__init__()
        self.rule_weight = rule_weight
        self.consistency_weight = consistency_weight
    
    def symbolic_rules(self, drug_hidden, protein_hidden, disease_hidden, knowledge_graph):
        rule_violations = 0
        total_rules = 0
        
        for drug_idx in range(min(5, knowledge_graph.num_drugs)):
            for disease_idx in range(min(5, knowledge_graph.num_diseases)):
                pathway_exists = any((protein_idx, disease_idx) in knowledge_graph.protein_to_disease for protein_idx in range(knowledge_graph.num_proteins))
                
                if pathway_exists:
                    drug_logits = torch.sigmoid(torch.sum(drug_hidden[drug_idx] * disease_hidden[disease_idx]))
                    rule_target = torch.tensor(0.8)
                    rule_violations += F.mse_loss(drug_logits, rule_target)
                    total_rules += 1
        
        return rule_violations / max(total_rules, 1)
    
    def consistency_loss(self, logits, uncertainty):
        probs = torch.sigmoid(logits)
        confidence = torch.abs(probs - 0.5) * 2
        inconsistency = confidence * uncertainty
        return torch.mean(inconsistency)
    
    def forward(self, positive_logits, negative_logits, positive_uncertainty, negative_uncertainty, drug_hidden, protein_hidden, disease_hidden, knowledge_graph):
        positive_loss = F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
        negative_loss = F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
        neural_loss = positive_loss + negative_loss
        
        rule_loss = self.symbolic_rules(drug_hidden, protein_hidden, disease_hidden, knowledge_graph)
        
        all_logits = torch.cat([positive_logits, negative_logits])
        all_uncertainty = torch.cat([positive_uncertainty, negative_uncertainty])
        consistency_loss = self.consistency_loss(all_logits, all_uncertainty)
        
        total_loss = neural_loss + self.rule_weight * rule_loss + self.consistency_weight * consistency_loss
        
        return total_loss, {
            'neural': neural_loss.item(),
            'rule': rule_loss.item() if isinstance(rule_loss, torch.Tensor) else rule_loss,
            'consistency': consistency_loss.item()
        }



def pair_tensor(pairs):
    if not pairs: 
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    pairs_array = np.array(pairs)
    return torch.tensor(pairs_array[:, 0], dtype=torch.long), torch.tensor(pairs_array[:, 1], dtype=torch.long)

def eval_model(model, knowledge_graph, split="test"):
    model.eval()
    positive_pairs, negative_pairs = (knowledge_graph.train_pos, knowledge_graph.train_neg) if split == "train" else \
                          (knowledge_graph.val_pos, knowledge_graph.val_neg) if split == "val" else (knowledge_graph.test_pos, knowledge_graph.test_neg)
    
    if not positive_pairs and not negative_pairs:
        return TrainingResult(0.0, 0.5, 0.5, 0.5, 0.5)
    
    drug_features = torch.tensor(knowledge_graph.drug_features, dtype=torch.float32)
    protein_features = torch.tensor(knowledge_graph.protein_features, dtype=torch.float32)
    disease_features = torch.tensor(knowledge_graph.disease_features, dtype=torch.float32)
    protein_disease_edges = torch.tensor(knowledge_graph.protein_to_disease, dtype=torch.long).t() if knowledge_graph.protein_to_disease else torch.empty(2, 0, dtype=torch.long)
    protein_protein_edges = torch.tensor(knowledge_graph.protein_to_protein, dtype=torch.long).t() if knowledge_graph.protein_to_protein else torch.empty(2, 0, dtype=torch.long)
    
    with torch.no_grad():
        drug_hidden, protein_hidden, disease_hidden = model(drug_features, protein_features, disease_features, protein_disease_edges, protein_protein_edges)
        
        if positive_pairs:
            positive_drugs, positive_diseases = pair_tensor(positive_pairs)
            positive_proteins = []
            for drug_idx, disease_idx in positive_pairs:
                related_proteins = [p_idx for p_idx, d_idx in knowledge_graph.protein_to_disease if d_idx == disease_idx]
                protein_idx = related_proteins[0] if related_proteins else 0
                positive_proteins.append(protein_idx)
            positive_protein_tensor = torch.tensor(positive_proteins, dtype=torch.long)
            positive_logits, positive_uncertainty = model.predict(drug_hidden[positive_drugs], disease_hidden[positive_diseases], protein_hidden[positive_protein_tensor])
        else:
            positive_logits = torch.empty(0)
            
        if negative_pairs:
            negative_drugs, negative_diseases = pair_tensor(negative_pairs)
            negative_proteins = []
            for drug_idx, disease_idx in negative_pairs:
                related_proteins = [p_idx for p_idx, d_idx in knowledge_graph.protein_to_disease if d_idx == disease_idx]
                protein_idx = related_proteins[0] if related_proteins else 0
                negative_proteins.append(protein_idx)
            negative_protein_tensor = torch.tensor(negative_proteins, dtype=torch.long)
            negative_logits, negative_uncertainty = model.predict(drug_hidden[negative_drugs], disease_hidden[negative_diseases], protein_hidden[negative_protein_tensor])
        else:
            negative_logits = torch.empty(0)
        
        if len(positive_logits) == 0 and len(negative_logits) == 0:
            return TrainingResult(0.0, 0.5, 0.5, 0.5, 0.5)
        
        all_probabilities = torch.cat([torch.sigmoid(positive_logits), torch.sigmoid(negative_logits)]).cpu().numpy()
        all_labels = np.concatenate([np.ones(len(positive_pairs)), np.zeros(len(negative_pairs))])
        
        if len(np.unique(all_labels)) < 2:
            return TrainingResult(0.0, 0.5, 0.5, (all_probabilities > 0.5).mean(), 0.5)
        
        area_under_curve = roc_auc_score(all_labels, all_probabilities)
        average_precision = average_precision_score(all_labels, all_probabilities)
        accuracy = ((all_probabilities > 0.5) == all_labels).mean()
        brier_score = brier_score_loss(all_labels, all_probabilities)
        
        loss = torch.tensor(0.0)
        if len(positive_logits) > 0:
            loss += F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
        if len(negative_logits) > 0:
            loss += F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
    
    return TrainingResult(loss.item(), area_under_curve, average_precision, accuracy, brier_score)

def train_epoch(model, knowledge_graph, optimizer):
    model.train()
    drug_features = torch.tensor(knowledge_graph.drug_features, dtype=torch.float32)
    protein_features = torch.tensor(knowledge_graph.protein_features, dtype=torch.float32)
    disease_features = torch.tensor(knowledge_graph.disease_features, dtype=torch.float32)
    protein_disease_edges = torch.tensor(knowledge_graph.protein_to_disease, dtype=torch.long).t() if knowledge_graph.protein_to_disease else torch.empty(2, 0, dtype=torch.long)
    protein_protein_edges = torch.tensor(knowledge_graph.protein_to_protein, dtype=torch.long).t() if knowledge_graph.protein_to_protein else torch.empty(2, 0, dtype=torch.long)
    
    drug_hidden, protein_hidden, disease_hidden = model(drug_features, protein_features, disease_features, protein_disease_edges, protein_protein_edges)
    
    batch_size = min(32, len(knowledge_graph.train_pos))
    positive_sample = random.sample(knowledge_graph.train_pos, batch_size)
    negative_sample = random.sample(knowledge_graph.train_neg, min(batch_size, len(knowledge_graph.train_neg)))
    
    positive_drugs, positive_diseases = pair_tensor(positive_sample)
    negative_drugs, negative_diseases = pair_tensor(negative_sample)
    
    # Get related proteins for each drug-disease pair to include in prediction
    positive_proteins = []
    for drug_idx, disease_idx in positive_sample:
        related_proteins = [p_idx for p_idx, d_idx in knowledge_graph.protein_to_disease if d_idx == disease_idx]
        if related_proteins:
            protein_idx = related_proteins[0]
            positive_proteins.append(protein_idx)
        else:
            positive_proteins.append(0)  # fallback to first protein
    
    negative_proteins = []
    for drug_idx, disease_idx in negative_sample:
        related_proteins = [p_idx for p_idx, d_idx in knowledge_graph.protein_to_disease if d_idx == disease_idx]
        if related_proteins:
            protein_idx = related_proteins[0]
            negative_proteins.append(protein_idx)
        else:
            negative_proteins.append(0)
    
    positive_protein_tensor = torch.tensor(positive_proteins, dtype=torch.long)
    negative_protein_tensor = torch.tensor(negative_proteins, dtype=torch.long)
    
    positive_logits, positive_uncertainty = model.predict(drug_hidden[positive_drugs], disease_hidden[positive_diseases], protein_hidden[positive_protein_tensor])
    negative_logits, negative_uncertainty = model.predict(drug_hidden[negative_drugs], disease_hidden[negative_diseases], protein_hidden[negative_protein_tensor])
    
    positive_loss = F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
    negative_loss = F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
    total_loss = positive_loss + negative_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return total_loss.item(), {'neural': total_loss.item(), 'rule': 0.0, 'consistency': 0.0}

def save_model(model, knowledge_graph, filepath):
    """Save trained model and knowledge graph"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'drug_input_dim': knowledge_graph.drug_dim,
            'protein_input_dim': knowledge_graph.protein_dim, 
            'disease_input_dim': knowledge_graph.disease_dim,
            'hidden_dim': 64  # Match the actual model architecture
        },
        'knowledge_graph_data': {
            'num_drugs': knowledge_graph.num_drugs, 
            'num_proteins': knowledge_graph.num_proteins, 
            'num_diseases': knowledge_graph.num_diseases,
            'node_names': knowledge_graph.node_names,
            'drug_features': knowledge_graph.drug_features, 
            'protein_features': knowledge_graph.protein_features, 
            'disease_features': knowledge_graph.disease_features,
            'protein_to_disease': knowledge_graph.protein_to_disease, 
            'protein_to_protein': knowledge_graph.protein_to_protein
        }
    }, filepath)
    print(f"Model saved to {filepath}")

def train_model(knowledge_graph, epochs=20, save_path="trained_causal_model.pth"):
    """Train causal model and save it"""
    model = CausalGNN(knowledge_graph.drug_dim, knowledge_graph.protein_dim, knowledge_graph.disease_dim, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    
    print("Training model...")
    best_area_under_curve = 0
    
    for epoch in range(epochs):
        train_loss, loss_components = train_epoch(model, knowledge_graph, optimizer)
        
        if epoch % 5 == 0:
            validation_result = eval_model(model, knowledge_graph, "val")
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val AUC={validation_result.auc:.4f}")
            
            if validation_result.auc > best_area_under_curve:
                best_area_under_curve = validation_result.auc
                save_model(model, knowledge_graph, save_path)
    
    test_result = eval_model(model, knowledge_graph, "test")
    print(f"\nFinal Model Performance:")
    print(f"Test AUC: {test_result.auc:.4f}")
    print(f"Test AP: {test_result.ap:.4f}")
    print(f"Test Accuracy: {test_result.acc:.4f}")
    
    return model, test_result

def main():
    pennsieve_path = os.environ.get("PENNSIEVE_DATA_DIR", DEFAULT_DATA_DIR)
    knowledge_graph = load_pennsieve(pennsieve_path, dimension=96)
    
    print(f"Loaded dataset: {knowledge_graph.num_drugs} compounds, {knowledge_graph.num_proteins} genes, {knowledge_graph.num_diseases} diseases")
    
    model, test_result = train_model(knowledge_graph, epochs=20, save_path="trained_causal_model.pth")
    print("Training completed. Model saved.")

if __name__ == "__main__":
    main()