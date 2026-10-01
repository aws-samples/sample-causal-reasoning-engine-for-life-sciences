import React, { useState } from 'react';
import { Search, Database, Layers } from 'lucide-react';

const NodeExplorer = ({ nodeData }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState('all');

  const nodeTypes = [
    { id: 'all', name: 'All Nodes', icon: Database },
    { id: 'compounds', name: 'Compounds', icon: Layers },
    { id: 'genes', name: 'Genes', icon: Layers },
    { id: 'diseases', name: 'Diseases', icon: Layers }
  ];

  const getFilteredNodes = () => {
    if (!nodeData) return [];
    
    let nodes = [];
    if (selectedType === 'all' || selectedType === 'compounds') {
      nodes = [...nodes, ...(nodeData.compounds || []).map(n => ({ ...n, type: 'compound' }))];
    }
    if (selectedType === 'all' || selectedType === 'genes') {
      nodes = [...nodes, ...(nodeData.genes || []).map(n => ({ ...n, type: 'gene' }))];
    }
    if (selectedType === 'all' || selectedType === 'diseases') {
      nodes = [...nodes, ...(nodeData.diseases || []).map(n => ({ ...n, type: 'disease' }))];
    }

    if (searchTerm) {
      nodes = nodes.filter(node => 
        node.name?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    return nodes.slice(0, 50); // Limit to 50 results
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'compound': return '#3b82f6';
      case 'gene': return '#10b981';
      case 'disease': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="node-explorer">
      <div className="explorer-header">
        <Database size={20} />
        <h3>Knowledge Graph Explorer</h3>
      </div>

      <div className="search-section">
        <div className="search-input">
          <Search size={16} />
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="type-filters">
          {nodeTypes.map(type => {
            const Icon = type.icon;
            return (
              <button
                key={type.id}
                className={`type-filter ${selectedType === type.id ? 'active' : ''}`}
                onClick={() => setSelectedType(type.id)}
              >
                <Icon size={14} />
                {type.name}
              </button>
            );
          })}
        </div>
      </div>

      <div className="nodes-list">
        {getFilteredNodes().map((node, idx) => (
          <div key={idx} className="node-item">
            <div 
              className="node-type-indicator"
              style={{ backgroundColor: getTypeColor(node.type) }}
            />
            <div className="node-info">
              <div className="node-name">{node.name || `${node.type}-${idx}`}</div>
              <div className="node-type">{node.type}</div>
            </div>
          </div>
        ))}
        {getFilteredNodes().length === 0 && (
          <div className="no-nodes">
            {nodeData ? 'No nodes found' : 'Loading nodes...'}
          </div>
        )}
      </div>
    </div>
  );
};

export default NodeExplorer;