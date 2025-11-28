# Backend Implementation Plan & File Structure

## 📁 Simplified File Structure

```
medical-ai-backend/
├── main.py                          # FastAPI entry point, API routes
├── config.py                        # Configuration management (DB, model paths, etc.)
├── requirements.txt                 # Python dependencies
│
├── models/
│   ├── vision_models.py            # ViT, CNN models for image analysis
│   ├── language_models.py          # BioGPT, Medical LLMs, RAG pipeline
│   ├── multimodal_fusion.py        # Vision-Language integration
│   └── continual_learning.py       # LoRA, adapter-based fine-tuning
│
├── knowledge/
│   ├── graph_rag.py                # Graph RAG implementation with Neo4j
│   ├── ontology_manager.py         # SNOMED, UMLS integration
│   ├── vector_store.py             # Embeddings storage and retrieval
│   └── knowledge_updater.py        # Dynamic knowledge ingestion
│
├── reasoning/
│   ├── fol_engine.py               # First-Order Logic inference
│   ├── srlm_reasoner.py            # Statistical Relational Learning Models
│   ├── protocol_validator.py       # Clinical protocol checking
│   └── chain_of_thought.py         # CoT reasoning generation
│
├── explainability/
│   ├── attention_viz.py            # Attention heatmaps, Grad-CAM
│   ├── feature_attribution.py      # SHAP, LIME implementations
│   ├── fol_proof_generator.py      # Logical proof explanations
│   └── report_generator.py         # Clinical report templates
│
├── data/
│   ├── image_processor.py          # DICOM handling, preprocessing
│   ├── text_processor.py           # Clinical text normalization
│   ├── dataset_loader.py           # CheXpert, MIMIC-CXR loaders
│   └── augmentation.py             # Data augmentation pipelines
│
├── services/
│   ├── inference_service.py        # Main prediction pipeline
│   ├── training_service.py         # Model training & fine-tuning
│   ├── knowledge_service.py        # Knowledge graph operations
│   └── explanation_service.py      # XAI pipeline orchestration
│
├── utils/
│   ├── medical_metrics.py          # AUC, F1, sensitivity, specificity
│   ├── visualization.py            # Plotting, heatmap generation
│   └── logging.py                  # Structured logging setup
│
├── database/
│   ├── postgres_client.py          # PostgreSQL + pgvector
│   ├── neo4j_client.py             # Neo4j graph operations
│   └── redis_client.py             # Caching layer
│
├── api/
│   ├── routes_radiology.py         # Radiology-specific endpoints
│   ├── routes_epidemiology.py      # Infectious disease endpoints
│   ├── routes_knowledge.py         # Knowledge update endpoints
│   └── middleware.py               # Auth, rate limiting, CORS
│
├── tests/
│   ├── test_models.py              # Model unit tests
│   ├── test_reasoning.py           # FOL/SRLM tests
│   └── test_api.py                 # API integration tests
│
├── docker/
│   ├── Dockerfile                  # Main application container
│   ├── docker-compose.yml          # Multi-service orchestration
│   └── .dockerignore
│
└── scripts/
    ├── download_models.py          # Download pretrained models
    ├── setup_knowledge_graph.py    # Initialize Neo4j with ontologies
    └── benchmark_evaluation.py     # Run CheXpert/MIMIC-CXR evaluation
```

---

## 🎯 Core Components Implementation Plan

### **1. Knowledge Layer: Graph RAG Implementation**

**File: `knowledge/graph_rag.py`**

```python
"""
Graph RAG combines:
- Medical knowledge graphs (Neo4j)
- Vector embeddings (pgvector)
- Semantic retrieval (Sentence Transformers)
"""

Key Components:
1. Entity extraction from medical text
2. Graph traversal for multi-hop reasoning
3. Hybrid search (vector similarity + graph structure)
4. Dynamic knowledge updates from WHO/CDC bulletins
```

**Integration Points:**
- Retrieves relevant medical knowledge for LLM context
- Grounds model predictions in authoritative sources
- Enables explainable reasoning through graph paths

---

### **2. Reasoning Layer: FOL + SRLM**

**File: `reasoning/fol_engine.py`**

```python
"""
First-Order Logic for clinical reasoning:
- Represents clinical rules as logical predicates
- Validates diagnostic conclusions
- Generates formal proofs
"""

Example FOL Rules:
- ∀x (Pneumonia(x) ∧ Fever(x) → Antibiotics_Indicated(x))
- ∀x (COVID_Positive(x) ∧ Oxygen_Sat(x) < 94 → Hospitalization_Required(x))
```

**File: `reasoning/srlm_reasoner.py`**

```python
"""
Statistical Relational Learning Models:
- Combines probabilistic reasoning with relational logic
- Handles uncertainty in medical diagnoses
- Learns from both structured rules and data patterns
"""

Use Cases:
- Uncertain disease relationships
- Risk stratification
- Treatment outcome prediction
```

**Integration:**
- FOL validates model outputs against clinical protocols
- SRLM handles probabilistic reasoning
- Both generate traceable reasoning chains for XAI

---

### **3. XAI Layer: Multi-faceted Explanations**

**File: `explainability/attention_viz.py`**

```python
"""
Visual explanations for image-based predictions:
- Grad-CAM for CNNs
- Attention rollout for Vision Transformers
- Segmentation overlays
"""
```

**File: `explainability/feature_attribution.py`**

```python
"""
Feature importance explanations:
- SHAP (Shapley values)
- LIME (local linear approximations)
- Integrated Gradients
"""
```

**File: `explainability/fol_proof_generator.py`**

```python
"""
Logical reasoning explanations:
- Converts FOL proofs to natural language
- Shows rule applications step-by-step
- Highlights violated/satisfied constraints
"""
```

**Unified Explanation Output:**
```json
{
  "prediction": "Pneumonia (Confidence: 0.87)",
  "visual_explanation": {
    "attention_heatmap": "base64_image_data",
    "key_regions": ["right lower lobe", "left costophrenic angle"]
  },
  "feature_importance": {
    "infiltrate_opacity": 0.42,
    "pleural_effusion": 0.28,
    "patient_fever": 0.18
  },
  "logical_reasoning": {
    "rule_chain": [
      "IF infiltrate_present AND fever > 38C THEN pneumonia_suspected",
      "IF pneumonia_suspected AND elevated_WBC THEN pneumonia_confirmed"
    ],
    "confidence_source": "Aligned with ACR Appropriateness Criteria 2024"
  },
  "knowledge_graph_path": [
    "Patient Symptoms → Pneumonia → Treatment Protocol → Antibiotics"
  ]
}
```

---

### **4. Multimodal Pipeline**

**File: `models/multimodal_fusion.py`**

```python
"""
Vision-Language Fusion Architecture:

1. Image Path:
   - ViT/Swin Transformer → Image embeddings [B, 197, 768]

2. Text Path:
   - BioGPT/Med-PaLM → Text embeddings [B, seq_len, 768]

3. Cross-Modal Attention:
   - Q_img = Linear(image_embeds)
   - K_text, V_text = Linear(text_embeds)
   - Attention(Q_img, K_text, V_text)

4. Fusion:
   - Concatenate aligned embeddings
   - Feed to classification head
"""

Models to Integrate:
- LLaVA-Med (pre-trained vision-language)
- BiomedCLIP (medical CLIP variant)
- Custom fusion with cross-attention
```

---

### **5. Continual Learning Pipeline**

**File: `models/continual_learning.py`**

```python
"""
Parameter-Efficient Fine-Tuning with LoRA:

1. Freeze base model weights
2. Inject low-rank adapters: ΔW = W_A × W_B
3. Train only adapter parameters (<<1% of total)
4. Merge adapters for inference

Benefits:
- Update with new protocols in hours (not weeks)
- Prevent catastrophic forgetting
- Minimal GPU requirements (1x A100 sufficient)
"""

Implementation:
- PEFT library for LoRA
- Replay buffer for old samples
- Elastic Weight Consolidation (EWC) for stability
```

---

## 🔧 Technology Stack Details

### **Core ML/DL**
```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
peft>=0.4.0  # Parameter-Efficient Fine-Tuning
timm>=0.9.0  # Vision models
monai>=1.2.0  # Medical imaging toolkit
```

### **Knowledge & RAG**
```python
langchain>=0.1.0
neo4j>=5.0.0
sentence-transformers>=2.2.0
pgvector>=0.1.0
chromadb>=0.4.0  # Vector database option
```

### **XAI & Reasoning**
```python
captum>=0.6.0  # PyTorch interpretability
shap>=0.42.0
lime>=0.2.0
problog>=2.2.0  # Probabilistic logic programming
pyke>=1.1.1  # Knowledge engine for FOL
networkx>=3.0  # Graph operations
```

### **Data & APIs**
```python
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydicom>=2.3.0  # DICOM processing
SimpleITK>=2.2.0
nibabel>=5.0.0  # NIfTI format
pandas>=2.0.0
numpy>=1.24.0
```

---

## 🚀 Implementation Phases

### **Phase 1: Foundation (Weeks 1-2)**
- [ ] Set up FastAPI project structure
- [ ] Configure PostgreSQL + Neo4j + Redis
- [ ] Implement data loaders (CheXpert, MIMIC-CXR)
- [ ] Basic image preprocessing pipeline
- [ ] Initial API endpoints

### **Phase 2: Model Integration (Weeks 3-4)**
- [ ] Load pretrained ViT for chest X-rays
- [ ] Integrate BioGPT for text processing
- [ ] Implement basic multimodal fusion
- [ ] LoRA fine-tuning pipeline
- [ ] Model serving infrastructure

### **Phase 3: Knowledge Layer (Weeks 5-6)**
- [ ] Build medical knowledge graph in Neo4j
- [ ] Implement Graph RAG retrieval
- [ ] Vector embedding storage (pgvector)
- [ ] Dynamic knowledge update pipeline
- [ ] SNOMED/UMLS integration

### **Phase 4: Reasoning (Weeks 7-8)**
- [ ] FOL rule engine implementation
- [ ] SRLM reasoner for probabilistic inference
- [ ] Clinical protocol validator
- [ ] Chain-of-thought reasoning module
- [ ] Integration with model predictions

### **Phase 5: XAI Layer (Weeks 9-10)**
- [ ] Attention visualization (Grad-CAM, rollout)
- [ ] SHAP/LIME feature attribution
- [ ] FOL proof generation
- [ ] Unified explanation API
- [ ] Report template system

### **Phase 6: Validation & Deployment (Weeks 11-12)**
- [ ] Benchmark evaluation (AUC, F1, sensitivity)
- [ ] Radiologist evaluation protocol
- [ ] Docker containerization
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Performance optimization

---

## 🧪 Key API Endpoints

```python
# Radiology Inference
POST /api/v1/radiology/predict
{
  "image": "base64_dicom_data",
  "clinical_context": "Patient presents with fever and cough",
  "modality": "chest_xray",
  "explain": true
}

# Knowledge Update
POST /api/v1/knowledge/update
{
  "source": "WHO Bulletin 2024-11",
  "content": "Updated COVID-19 variant classification",
  "type": "guideline"
}

# Continual Learning
POST /api/v1/models/fine-tune
{
  "dataset": "new_protocol_samples",
  "method": "lora",
  "base_model": "biogpt-large"
}

# Explanation Generation
GET /api/v1/explain/prediction/{prediction_id}
{
  "visual": true,
  "logical_proof": true,
  "feature_importance": true,
  "knowledge_path": true
}
```

---

## 📊 Expected Performance Metrics

| Component | Metric | Target |
|-----------|--------|--------|
| Radiology Classification | AUC-ROC | >0.90 |
| Report Generation | BLEU-4 | >0.35 |
| Knowledge Retrieval | Recall@5 | >0.85 |
| FOL Reasoning | Rule Coverage | >90% |
| XAI Quality | Radiologist Agreement | >80% |
| Inference Latency | Response Time | <3s |

---

## 🔒 Critical Considerations

### **Security & Privacy**
- HIPAA compliance for medical data
- PHI de-identification pipeline
- Encrypted storage and transmission
- Role-based access control (RBAC)

### **Scalability**
- Async processing for heavy tasks
- Redis queue for batch inference
- Model versioning and A/B testing
- Horizontal scaling with Kubernetes

### **Clinical Integration**
- HL7/FHIR compatibility
- PACS integration (DICOM endpoints)
- EHR webhook support
- Audit logging for regulatory compliance

---

This architecture provides a complete, production-ready backend that integrates cutting-edge AI research (Graph RAG, FOL reasoning, SRLM, XAI) with practical clinical deployment requirements. The simplified file structure keeps related functionality together while maintaining clear separation of concerns.