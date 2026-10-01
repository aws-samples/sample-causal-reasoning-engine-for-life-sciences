import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, NamedTuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)

class TrainingResult(NamedTuple):
    loss: float
    auc: float
    ap: float
    acc: float
    brier: float

@dataclass
class HeterogeneousKnowledgeGraph:
    graph: nx.DiGraph
    num_drugs: int
    num_proteins: int
    num_diseases: int
    drug_dim: int
    protein_dim: int
    disease_dim: int
    drug_features: np.ndarray
    protein_features: np.ndarray
    disease_features: np.ndarray
    drug_to_protein_inhibition: List[Tuple[int, int]]
    drug_to_protein_activation: List[Tuple[int, int]]
    protein_to_disease: List[Tuple[int, int]]
    protein_to_protein: List[Tuple[int, int]]
    node_mappings: Dict[str, Dict[str, int]]
    node_names: Dict[str, Dict[int, str]]
    train_pos: List[Tuple[int, int]]
    train_neg: List[Tuple[int, int]]
    val_pos: List[Tuple[int, int]]
    val_neg: List[Tuple[int, int]]
    test_pos: List[Tuple[int, int]]
    test_neg: List[Tuple[int, int]]
    envs: List[Dict[str, Any]]
    env_assign: Dict[str, List[int]]

class CausalGNN(nn.Module):
    def __init__(self, drug_input_dim: int, protein_input_dim: int, disease_input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.drug_projection = nn.Linear(drug_input_dim, hidden_dim)
        self.protein_projection = nn.Linear(protein_input_dim, hidden_dim)
        self.disease_projection = nn.Linear(disease_input_dim, hidden_dim)
        self.graph_neural_network_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.protein_protein_layer = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear_predictor = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim * 2, 1)
        self.uncertainty_head_with_protein = nn.Linear(hidden_dim * 3, 1)
        self.dropout = nn.Dropout(0.1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, drug_features, protein_features, disease_features, protein_disease_edges, protein_protein_edges=None):
        drug_hidden = F.relu(self.drug_projection(drug_features))
        protein_hidden = F.relu(self.protein_projection(protein_features))
        disease_hidden = F.relu(self.disease_projection(disease_features))
        
        # Protein-protein interactions
        if protein_protein_edges is not None and protein_protein_edges.size(1) > 0:
            protein_source, protein_target = protein_protein_edges[0], protein_protein_edges[1]
            protein_messages = torch.zeros_like(protein_hidden)
            protein_messages.index_add_(0, protein_target, protein_hidden[protein_source])
            protein_hidden = protein_hidden + F.relu(self.protein_protein_layer(self.dropout(protein_messages)))
        
        # Protein-disease interactions
        for layer in self.graph_neural_network_layers:
            if protein_disease_edges.size(1) > 0:
                protein_hidden_new, disease_hidden_new = protein_hidden.clone(), disease_hidden.clone()
                protein_indices, disease_indices = protein_disease_edges[0], protein_disease_edges[1]
                protein_hidden_new.index_add_(0, protein_indices, self.dropout(layer(disease_hidden[disease_indices])))
                disease_hidden_new.index_add_(0, disease_indices, self.dropout(layer(protein_hidden[protein_indices])))
                protein_hidden, disease_hidden = protein_hidden_new, disease_hidden_new
        
        return drug_hidden, protein_hidden, disease_hidden
    
    def predict(self, drug_embedding, disease_embedding, protein_embedding=None):
        if protein_embedding is not None:
            combined_embedding = torch.cat([drug_embedding, protein_embedding, disease_embedding], dim=-1)
            drug_proj = drug_embedding + 0.3 * protein_embedding
            disease_proj = disease_embedding + 0.3 * protein_embedding
            logits = self.bilinear_predictor(drug_proj, disease_proj).squeeze(-1)
            uncertainty = torch.sigmoid(self.uncertainty_head_with_protein(combined_embedding)).squeeze(-1)
        else:
            logits = self.bilinear_predictor(drug_embedding, disease_embedding).squeeze(-1)
            uncertainty = torch.sigmoid(self.uncertainty_head(torch.cat([drug_embedding, disease_embedding], dim=-1))).squeeze(-1)
        return logits, uncertainty



def load_pennsieve(data_directory: str, dimension: int = 96) -> HeterogeneousKnowledgeGraph:
    """Load Pennsieve dataset from TSV files"""
    try:
        dfN = pd.read_csv(os.path.join(data_directory, "nodes.tsv"), 
                         sep='\t', on_bad_lines='skip', engine='python')
        dfE = pd.read_csv(os.path.join(data_directory, "edges.sif"), 
                         sep='\t', on_bad_lines='skip', engine='python')
        
        id2kind = dict(zip(dfN.id, dfN.kind))
        id2name = dict(zip(dfN.id, dfN.name))
        
        all_nodes = {
            "Compound": [i for i, k in id2kind.items() if k == "Compound"][:100],
            "Gene": [i for i, k in id2kind.items() if k == "Gene"][:200], 
            "Disease": [i for i, k in id2kind.items() if k == "Disease"][:100],
            "Anatomy": [i for i, k in id2kind.items() if k == "Anatomy"][:100],
            "Biological Process": [i for i, k in id2kind.items() if k == "Biological Process"][:100],
            "Molecular Function": [i for i, k in id2kind.items() if k == "Molecular Function"][:100],
            "Side Effect": [i for i, k in id2kind.items() if k == "Side Effect"][:100],
            "Symptom": [i for i, k in id2kind.items() if k == "Symptom"][:100]
        }
        
        drug_ids, protein_ids, disease_ids = all_nodes["Compound"], all_nodes["Gene"], all_nodes["Disease"]
            
    except Exception as e:
        print(f"Error reading TSV files: {e}")
        raise e
    
    drug_mapping = {v: i for i, v in enumerate(drug_ids)}
    protein_mapping = {v: i for i, v in enumerate(protein_ids)}
    disease_mapping = {v: i for i, v in enumerate(disease_ids)}
    
    node_mappings = {}
    node_names = {}
    for node_type, node_ids in all_nodes.items():
        node_mappings[node_type] = {node_id: i for i, node_id in enumerate(node_ids)}
        node_names[node_type] = {i: id2name.get(node_id, f"{node_type}-{i}") for i, node_id in enumerate(node_ids)}
    
    drug_features = np.random.randn(len(drug_mapping), dimension).astype(np.float32)
    protein_features = np.random.randn(len(protein_mapping), dimension).astype(np.float32)
    disease_features = np.random.randn(len(disease_mapping), dimension).astype(np.float32)
    
    graph = nx.DiGraph()
    for i in range(len(drug_mapping)): graph.add_node(("drug_features", i), type="drug_features", x=drug_features[i])
    for i in range(len(protein_mapping)): graph.add_node(("protein_features", i), type="protein_features", x=protein_features[i])
    for i in range(len(disease_mapping)): graph.add_node(("disease_features", i), type="disease_features", x=disease_features[i])
    
    protein_to_disease = []
    protein_to_protein = []
    for _, row in dfE.iterrows():
        source, relation, target = row["source"], row["metaedge"], row["target"]
        if target in protein_mapping and source in disease_mapping and relation in {"DaG", "DuG", "DdG"}:
            protein_to_disease.append((protein_mapping[target], disease_mapping[source]))
            graph.add_edge(("protein_features", protein_mapping[target]), ("disease_features", disease_mapping[source]), rel=relation)
        elif source in protein_mapping and target in protein_mapping and relation in {"GiG", "Gr>G", "GcG"}:
            protein_to_protein.append((protein_mapping[source], protein_mapping[target]))
            graph.add_edge(("protein_features", protein_mapping[source]), ("protein_features", protein_mapping[target]), rel=relation)
    
    positive_pairs = []
    for _, row in dfE.iterrows():
        source, relation, target = row["source"], row["metaedge"], row["target"]
        if relation == "CtD" and source in drug_mapping and target in disease_mapping:
            positive_pairs.append((drug_mapping[source], disease_mapping[target]))
    
    if len(positive_pairs) < 20:
        compound_to_genes = {}
        gene_to_diseases = {}
        
        for _, row in dfE.iterrows():
            source, relation, target = row["source"], row["metaedge"], row["target"]
            if source in drug_mapping and target in protein_mapping and relation in {"CdG", "CuG", "CbG"}:
                if drug_mapping[source] not in compound_to_genes:
                    compound_to_genes[drug_mapping[source]] = []
                compound_to_genes[drug_mapping[source]].append(protein_mapping[target])
            if source in protein_mapping and target in disease_mapping and relation in {"DaG", "DuG", "DdG"}:
                if protein_mapping[source] not in gene_to_diseases:
                    gene_to_diseases[protein_mapping[source]] = []
                gene_to_diseases[protein_mapping[source]].append(disease_mapping[target])
        
        for compound_index, genes in compound_to_genes.items():
            for gene_index in genes:
                if gene_index in gene_to_diseases:
                    for disease_index in gene_to_diseases[gene_index]:
                        positive_pairs.append((compound_index, disease_index))
    
    positive_pairs = list(set(positive_pairs))
    all_pairs = [(d, x) for d in range(len(drug_mapping)) for x in range(len(disease_mapping))]
    negative_pairs = [pair for pair in all_pairs if pair not in positive_pairs]
    if len(negative_pairs) > len(positive_pairs):
        negative_pairs = random.sample(negative_pairs, len(positive_pairs))
    
    def split_data(data_list):
        random.shuffle(data_list)
        n = len(data_list)
        if n < 6:
            return data_list[:max(1, n//2)], data_list[max(1, n//2):max(2, 2*n//3)], data_list[max(2, 2*n//3):]
        train_size, validation_size = max(2, int(0.7 * n)), max(1, int(0.15 * n))
        return data_list[:train_size], data_list[train_size:train_size+validation_size], data_list[train_size+validation_size:]
    
    train_positive, validation_positive, test_positive = split_data(positive_pairs)
    train_negative, validation_negative, test_negative = split_data(negative_pairs)
    
    envs = [{"shift": np.random.randn(dimension).astype(np.float32) * 0.3 * (e + 1)} for e in range(3)]
    env_assign = {"train": [0, 1], "val": [2], "test": [2]}
    
    return HeterogeneousKnowledgeGraph(graph, len(drug_mapping), len(protein_mapping), len(disease_mapping), 
                   dimension, dimension, dimension, drug_features, protein_features, disease_features, [], [], 
                   protein_to_disease, protein_to_protein, node_mappings, node_names, 
                   train_positive, train_negative, validation_positive, validation_negative, 
                   test_positive, test_negative, envs, env_assign)