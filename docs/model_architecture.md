# CausalLLM Model Architecture

## Overview
The CausalLLM system implements a Causal Graph Neural Network (CausalGNN) for biomedical causal reasoning and drug repurposing.

## Core Architecture Diagram

```mermaid
graph TB
    subgraph InputLayer["Input Layer"]
        DRUG_FEATURES["Drug Features<br/>(drug_input_dim)"]
        PROTEIN_FEATURES["Protein Features<br/>(protein_input_dim)"]
        DISEASE_FEATURES["Disease Features<br/>(disease_input_dim)"]
    end
    
    subgraph KnowledgeGraph["Knowledge Graph Structure"]
        PROTEIN_DISEASE["Protein-Disease Edges<br/>(protein_disease_edges)"]
        PROTEIN_PROTEIN["Protein-Protein Edges<br/>(protein_protein_edges)"]
    end
    
    subgraph CausalGNNCore["CausalGNN Core Model"]
        subgraph ProjectionLayers["Projection Layers"]
            DRUG_PROJ["Drug Projection<br/>Linear(drug_dim → hidden_dim)"]
            PROTEIN_PROJ["Protein Projection<br/>Linear(protein_dim → hidden_dim)"]
            DISEASE_PROJ["Disease Projection<br/>Linear(disease_dim → hidden_dim)"]
        end
        
        subgraph GraphLayers["Graph Neural Network Layers"]
            PP_LAYER["Protein-Protein Layer<br/>Linear(hidden_dim → hidden_dim)"]
            GNN_LAYER1["GNN Layer 1<br/>Linear(hidden_dim → hidden_dim)"]
            GNN_LAYER2["GNN Layer 2<br/>Linear(hidden_dim → hidden_dim)"]
            DROPOUT["Dropout(0.1)"]
        end
        
        subgraph OutputRep["Output Representations"]
            DRUG_HIDDEN["Drug Hidden<br/>(hidden_dim)"]
            PROTEIN_HIDDEN["Protein Hidden<br/>(hidden_dim)"]
            DISEASE_HIDDEN["Disease Hidden<br/>(hidden_dim)"]
        end
    end
    
    subgraph PredictionModule["Prediction Module"]
        BILINEAR_PREDICTOR["Bilinear Predictor<br/>Bilinear(hidden_dim, hidden_dim, 1)"]
        UNCERTAINTY_HEAD["Uncertainty Head<br/>Linear(hidden_dim×2, 1)"]
        UNCERTAINTY_HEAD_PROTEIN["Uncertainty Head w/ Protein<br/>Linear(hidden_dim×3, 1)"]
        PREDICTION_LOGITS["Prediction Logits"]
        UNCERTAINTY_SCORES["Uncertainty Scores"]
    end
    
    subgraph NeuroSymbolic["Neuro-Symbolic Components"]
        SYMBOLIC_RULES["Symbolic Rules<br/>Domain Knowledge"]
        NEURAL_LOSS["Neural Loss<br/>BCE Loss"]
        RULE_LOSS["Rule Loss<br/>Symbolic Constraints"]
        CONSISTENCY_LOSS["Consistency Loss<br/>Uncertainty Regularization"]
        TOTAL_LOSS["Total Loss<br/>Neural + Rule + Consistency"]
    end
    
    %% Data Flow
    DRUG_FEATURES --> DRUG_PROJ
    PROTEIN_FEATURES --> PROTEIN_PROJ
    DISEASE_FEATURES --> DISEASE_PROJ
    
    DRUG_PROJ --> DRUG_HIDDEN
    PROTEIN_PROJ --> PROTEIN_HIDDEN
    DISEASE_PROJ --> DISEASE_HIDDEN
    
    %% Protein-Protein Interactions
    PROTEIN_PROTEIN --> PP_LAYER
    PROTEIN_HIDDEN --> PP_LAYER
    PP_LAYER --> PROTEIN_HIDDEN
    
    %% Protein-Disease Message Passing
    PROTEIN_DISEASE --> GNN_LAYER1
    PROTEIN_HIDDEN --> GNN_LAYER1
    DISEASE_HIDDEN --> GNN_LAYER1
    GNN_LAYER1 --> DROPOUT
    DROPOUT --> GNN_LAYER2
    GNN_LAYER2 --> PROTEIN_HIDDEN
    GNN_LAYER2 --> DISEASE_HIDDEN
    
    %% Prediction
    DRUG_HIDDEN --> BILINEAR_PREDICTOR
    DISEASE_HIDDEN --> BILINEAR_PREDICTOR
    PROTEIN_HIDDEN --> BILINEAR_PREDICTOR
    
    DRUG_HIDDEN --> UNCERTAINTY_HEAD
    DISEASE_HIDDEN --> UNCERTAINTY_HEAD
    PROTEIN_HIDDEN --> UNCERTAINTY_HEAD_PROTEIN
    
    BILINEAR_PREDICTOR --> PREDICTION_LOGITS
    UNCERTAINTY_HEAD --> UNCERTAINTY_SCORES
    UNCERTAINTY_HEAD_PROTEIN --> UNCERTAINTY_SCORES
    
    %% Neuro-Symbolic Loss
    PREDICTION_LOGITS --> NEURAL_LOSS
    DRUG_HIDDEN --> RULE_LOSS
    PROTEIN_HIDDEN --> RULE_LOSS
    DISEASE_HIDDEN --> RULE_LOSS
    SYMBOLIC_RULES --> RULE_LOSS
    PREDICTION_LOGITS --> CONSISTENCY_LOSS
    UNCERTAINTY_SCORES --> CONSISTENCY_LOSS
    NEURAL_LOSS --> TOTAL_LOSS
    RULE_LOSS --> TOTAL_LOSS
    CONSISTENCY_LOSS --> TOTAL_LOSS
    
    %% Styling
    classDef input fill:#e1f5fe
    classDef projection fill:#f3e5f5
    classDef gnnlayer fill:#e8f5e8
    classDef output fill:#fff3e0
    classDef prediction fill:#fce4ec
    classDef neurosymbolic fill:#fff8e1
    
    class DRUG_FEATURES,PROTEIN_FEATURES,DISEASE_FEATURES input
    class DRUG_PROJ,PROTEIN_PROJ,DISEASE_PROJ projection
    class PP_LAYER,GNN_LAYER1,GNN_LAYER2,DROPOUT gnnlayer
    class DRUG_HIDDEN,PROTEIN_HIDDEN,DISEASE_HIDDEN output
    class BILINEAR_PREDICTOR,UNCERTAINTY_HEAD,UNCERTAINTY_HEAD_PROTEIN,PREDICTION_LOGITS,UNCERTAINTY_SCORES prediction
    class SYMBOLIC_RULES,NEURAL_LOSS,RULE_LOSS,CONSISTENCY_LOSS,TOTAL_LOSS neurosymbolic
```

## Model Components Detail

### 1. Input Features
- **Drug Features**: Chemical properties, molecular descriptors (configurable dimension)
- **Protein Features**: Protein sequences, functional annotations (configurable dimension)
- **Disease Features**: Clinical phenotypes, disease ontology (configurable dimension)

### 2. Graph Structure
- **protein_disease_edges**: Protein-disease associations from knowledge graph
- **protein_protein_edges**: Protein-protein regulatory networks and interactions

### 3. CausalGNN Architecture
```
Input Dimensions:
├── Drug: drug_input_dim (configurable)
├── Protein: protein_input_dim (configurable)
└── Disease: disease_input_dim (configurable)

Hidden Dimension: hidden_dim (default: 128)

Layers:
├── Projection: Linear(input_dim → hidden_dim) with ReLU
├── Protein-Protein: Linear(hidden_dim → hidden_dim) for P-P interactions
├── GNN Layers: 2 layers with message passing and dropout
└── Output: hidden_dim-dimensional representations
```

### 4. Prediction Mechanism
```python
# Bilinear prediction with optional protein integration
if protein_embedding is not None:
    drug_proj = drug_embedding + 0.3 * protein_embedding
    disease_proj = disease_embedding + 0.3 * protein_embedding
    logits = bilinear_predictor(drug_proj, disease_proj)
    uncertainty = uncertainty_head_with_protein(cat([drug, protein, disease]))
else:
    logits = bilinear_predictor(drug_embedding, disease_embedding)
    uncertainty = uncertainty_head(cat([drug_embedding, disease_embedding]))
```

### 5. Key Implementation Details
- **Xavier Initialization**: All linear layers use Xavier uniform initialization
- **Dropout**: 0.1 dropout rate applied during message passing
- **Message Passing**: Uses index_add_ for efficient sparse operations
- **Uncertainty Quantification**: Sigmoid activation for uncertainty scores

## Training Pipeline

```mermaid
graph LR
    subgraph "Data Preparation"
        KNOWLEDGE_GRAPH[Knowledge Graph<br/>Construction]
        DATA_SPLIT[Train/Val/Test<br/>Split]
    end
    
    subgraph "Training Loop"
        BATCH_SAMPLING[Batch Sampling<br/>Positive/Negative Pairs]
        FORWARD_PASS[Forward Pass<br/>CausalGNN]
        LOSS_CALCULATION[Loss Calculation]
        BACKPROPAGATION[Backpropagation]
    end
    
    subgraph "Evaluation"
        PERFORMANCE_METRICS[AUC, AP, Accuracy<br/>Brier Score]
        MODEL_CHECKPOINT[Model Checkpointing]
    end
    
    KNOWLEDGE_GRAPH --> DATA_SPLIT
    DATA_SPLIT --> BATCH_SAMPLING
    BATCH_SAMPLING --> FORWARD_PASS
    FORWARD_PASS --> LOSS_CALCULATION
    LOSS_CALCULATION --> BACKPROPAGATION
    BACKPROPAGATION --> BATCH_SAMPLING
    FORWARD_PASS --> PERFORMANCE_METRICS
    PERFORMANCE_METRICS --> MODEL_CHECKPOINT
```

## Key Features

### Heterogeneous Graph Processing
- Handles different node types (drugs, proteins, diseases) with separate projections
- Supports multiple edge types with specialized message passing
- Efficient sparse tensor operations for large-scale graphs

### Uncertainty-Aware Predictions
- Dual uncertainty heads for different prediction contexts
- Protein-aware uncertainty estimation when mediator information is available
- Calibrated uncertainty scores for reliable confidence estimates

### Scalability
- Memory-optimized adjacency matrix representations
- Batch processing for large-scale datasets
- Efficient graph convolution with index-based operations

## Performance Metrics

The model is evaluated using:
- **Area Under Curve (AUC)**: Area Under ROC Curve
- **Average Precision (AP)**: Average Precision Score
- **Accuracy**: Classification accuracy at 0.5 threshold
- **Brier Score**: Calibration metric for probabilistic predictions

## Data Structure

The model uses the `HeterogeneousKnowledgeGraph` dataclass containing:
- Node features for each entity type
- Edge lists for different relationship types
- Train/validation/test splits
- Environment assignments for causal inference
- Node mappings and names for interpretability