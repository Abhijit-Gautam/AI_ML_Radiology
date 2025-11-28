import React, { useState } from 'react';
import { Database, Cpu, Network, FileText, Activity, Shield, Layers, GitBranch } from 'lucide-react';

const ArchitectureDiagram = () => {
  const [selectedComponent, setSelectedComponent] = useState(null);

  const components = {
    'Data Layer': {
      icon: Database,
      color: 'bg-blue-500',
      items: ['Medical Images (DICOM)', 'Clinical Reports', 'Knowledge Graphs', 'Embeddings Store']
    },
    'Knowledge Layer': {
      icon: Network,
      color: 'bg-green-500',
      items: ['Graph RAG Engine', 'Medical Ontologies (SNOMED, UMLS)', 'FOL Rule Base', 'Dynamic Knowledge Base']
    },
    'Model Layer': {
      icon: Cpu,
      color: 'bg-purple-500',
      items: ['Vision Transformers', 'Medical LLMs (BioGPT)', 'Multimodal Fusion', 'Continual Learning Engine']
    },
    'Reasoning Layer': {
      icon: GitBranch,
      color: 'bg-orange-500',
      items: ['FOL Inference Engine', 'SRLM Reasoner', 'Chain-of-Thought Generator', 'Clinical Protocol Checker']
    },
    'XAI Layer': {
      icon: Activity,
      color: 'bg-red-500',
      items: ['Attention Visualizer', 'SHAP/LIME Explainer', 'FOL Proof Tracer', 'Report Generator']
    },
    'API Layer': {
      icon: Shield,
      color: 'bg-indigo-500',
      items: ['REST API', 'WebSocket (Real-time)', 'Authentication', 'Rate Limiting']
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">Expandable Medical AI Backend Architecture</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {Object.entries(components).map(([name, { icon: Icon, color, items }]) => (
          <div
            key={name}
            onClick={() => setSelectedComponent(name)}
            className={`p-4 rounded-lg shadow-lg cursor-pointer transition-all hover:scale-105 ${
              selectedComponent === name ? 'ring-4 ring-blue-400' : ''
            } bg-white`}
          >
            <div className="flex items-center mb-3">
              <div className={`p-2 rounded ${color}`}>
                <Icon className="text-white" size={24} />
              </div>
              <h3 className="ml-3 font-semibold text-lg">{name}</h3>
            </div>
            <ul className="space-y-1 text-sm text-gray-600">
              {items.map((item, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="mr-2">•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {selectedComponent && (
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-4">{selectedComponent} Details</h2>
          <ComponentDetails component={selectedComponent} />
        </div>
      )}

      <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Data Flow Architecture</h2>
        <DataFlowDiagram />
      </div>

      <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold mb-4">Technology Stack Summary</h2>
        <TechStack />
      </div>
    </div>
  );
};

const ComponentDetails = ({ component }) => {
  const details = {
    'Data Layer': {
      description: 'Handles all data ingestion, storage, and retrieval operations',
      tech: ['PostgreSQL + pgvector', 'Neo4j (Knowledge Graphs)', 'MinIO/S3 (Images)', 'Redis (Caching)'],
      integration: 'Connects to Knowledge Layer for semantic search and Model Layer for training data'
    },
    'Knowledge Layer': {
      description: 'Manages medical knowledge graphs and enables dynamic knowledge retrieval',
      tech: ['Neo4j for Graph Database', 'LangChain for RAG', 'NetworkX for graph operations', 'RDFLib for ontologies'],
      integration: 'Graph RAG retrieves contextual information for LLM inference and FOL reasoning'
    },
    'Model Layer': {
      description: 'Core AI models for vision, language, and multimodal understanding',
      tech: ['PyTorch', 'HuggingFace Transformers', 'timm (Vision Models)', 'PEFT/LoRA for fine-tuning'],
      integration: 'Receives data from Data Layer, enriched context from Knowledge Layer, outputs to XAI Layer'
    },
    'Reasoning Layer': {
      description: 'Provides logical reasoning, protocol validation, and structured inference',
      tech: ['ProbLog/PyKE (FOL)', 'Custom SRLM implementation', 'Prolog interface', 'Rule engine'],
      integration: 'Validates model outputs against clinical protocols, generates formal proofs for XAI'
    },
    'XAI Layer': {
      description: 'Generates human-interpretable explanations for model decisions',
      tech: ['Captum', 'SHAP', 'LIME', 'Custom attention visualizers', 'Template-based report generation'],
      integration: 'Combines model activations, FOL proofs, and knowledge graph paths into unified explanations'
    },
    'API Layer': {
      description: 'Exposes backend functionality through secure, scalable APIs',
      tech: ['FastAPI', 'WebSocket (Socket.io)', 'JWT Authentication', 'Redis rate limiting'],
      integration: 'Orchestrates all backend layers and serves frontend/external systems'
    }
  };

  const info = details[component];
  
  return (
    <div className="space-y-4">
      <p className="text-gray-700">{info.description}</p>
      
      <div>
        <h3 className="font-semibold mb-2">Technologies:</h3>
        <div className="flex flex-wrap gap-2">
          {info.tech.map((tech, idx) => (
            <span key={idx} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
              {tech}
            </span>
          ))}
        </div>
      </div>

      <div>
        <h3 className="font-semibold mb-2">Integration Points:</h3>
        <p className="text-gray-600 text-sm">{info.integration}</p>
      </div>
    </div>
  );
};

const DataFlowDiagram = () => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between p-4 bg-gray-50 rounded">
        <div className="text-center flex-1">
          <div className="font-semibold">Input</div>
          <div className="text-sm text-gray-600">Medical Image + Clinical Text</div>
        </div>
        <div className="text-2xl">→</div>
        <div className="text-center flex-1">
          <div className="font-semibold">Preprocessing</div>
          <div className="text-sm text-gray-600">DICOM parsing, Text normalization</div>
        </div>
        <div className="text-2xl">→</div>
        <div className="text-center flex-1">
          <div className="font-semibold">Graph RAG</div>
          <div className="text-sm text-gray-600">Knowledge retrieval</div>
        </div>
      </div>

      <div className="flex items-center justify-between p-4 bg-gray-50 rounded">
        <div className="text-center flex-1">
          <div className="font-semibold">Multimodal Fusion</div>
          <div className="text-sm text-gray-600">ViT + Medical LLM</div>
        </div>
        <div className="text-2xl">→</div>
        <div className="text-center flex-1">
          <div className="font-semibold">FOL Reasoning</div>
          <div className="text-sm text-gray-600">Protocol validation</div>
        </div>
        <div className="text-2xl">→</div>
        <div className="text-center flex-1">
          <div className="font-semibold">XAI Generation</div>
          <div className="text-sm text-gray-600">Explanations + Proofs</div>
        </div>
      </div>

      <div className="flex items-center justify-center p-4 bg-green-50 rounded">
        <div className="text-center">
          <div className="font-semibold text-green-800">Output</div>
          <div className="text-sm text-gray-600">Diagnosis + Confidence + Visual Explanations + Logical Proof</div>
        </div>
      </div>
    </div>
  );
};

const TechStack = () => {
  const stack = [
    { category: 'Core Framework', items: ['FastAPI (Python 3.11+)', 'Pydantic (Data validation)'] },
    { category: 'ML/DL', items: ['PyTorch 2.0+', 'HuggingFace Transformers', 'timm', 'MONAI'] },
    { category: 'Knowledge & RAG', items: ['LangChain', 'Neo4j', 'pgvector', 'Sentence Transformers'] },
    { category: 'XAI', items: ['Captum', 'SHAP', 'LIME', 'PyTorch hooks for attention'] },
    { category: 'FOL/SRLM', items: ['ProbLog', 'PyKE', 'NetworkX', 'Custom rule engine'] },
    { category: 'Storage', items: ['PostgreSQL', 'Neo4j', 'Redis', 'MinIO'] },
    { category: 'Deployment', items: ['Docker', 'Kubernetes (optional)', 'NGINX', 'Prometheus'] }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {stack.map((section, idx) => (
        <div key={idx} className="border rounded p-4">
          <h3 className="font-semibold mb-2 text-blue-700">{section.category}</h3>
          <ul className="space-y-1 text-sm text-gray-600">
            {section.items.map((item, i) => (
              <li key={i} className="flex items-start">
                <span className="mr-2">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
};

export default ArchitectureDiagram;