import React, { useEffect, useRef } from 'react';

const KnowledgeGraphVisualization = ({ graphData }) => {
  const svgRef = useRef();

  useEffect(() => {
    if (!graphData || !graphData.nodes || !graphData.edges) return;

    const svg = svgRef.current;
    const width = 400;
    const height = 300;

    // Clear previous content
    svg.innerHTML = '';

    // Create SVG elements
    const svgElement = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgElement.setAttribute('width', width);
    svgElement.setAttribute('height', height);
    svgElement.setAttribute('viewBox', `0 0 ${width} ${height}`);

    // Simple force-directed layout simulation
    const nodes = graphData.nodes.map(node => ({
      ...node,
      x: Math.random() * (width - 100) + 50,
      y: Math.random() * (height - 100) + 50,
      vx: 0,
      vy: 0
    }));

    const edges = graphData.edges;

    // Simple physics simulation
    for (let i = 0; i < 100; i++) {
      // Repulsion between nodes
      for (let j = 0; j < nodes.length; j++) {
        for (let k = j + 1; k < nodes.length; k++) {
          const dx = nodes[k].x - nodes[j].x;
          const dy = nodes[k].y - nodes[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = 500 / (distance * distance);
          const fx = (dx / distance) * force;
          const fy = (dy / distance) * force;
          nodes[j].vx -= fx;
          nodes[j].vy -= fy;
          nodes[k].vx += fx;
          nodes[k].vy += fy;
        }
      }

      // Attraction along edges
      edges.forEach(edge => {
        const source = nodes.find(n => n.id === edge.source);
        const target = nodes.find(n => n.id === edge.target);
        if (source && target) {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = distance * 0.01;
          const fx = (dx / distance) * force;
          const fy = (dy / distance) * force;
          source.vx += fx;
          source.vy += fy;
          target.vx -= fx;
          target.vy -= fy;
        }
      });

      // Update positions
      nodes.forEach(node => {
        node.vx *= 0.9;
        node.vy *= 0.9;
        node.x += node.vx;
        node.y += node.vy;
        node.x = Math.max(20, Math.min(width - 20, node.x));
        node.y = Math.max(20, Math.min(height - 20, node.y));
      });
    }

    // Draw edges
    edges.forEach(edge => {
      const source = nodes.find(n => n.id === edge.source);
      const target = nodes.find(n => n.id === edge.target);
      if (source && target) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', source.x);
        line.setAttribute('y1', source.y);
        line.setAttribute('x2', target.x);
        line.setAttribute('y2', target.y);
        line.setAttribute('stroke', getEdgeColor(edge.type));
        line.setAttribute('stroke-width', '2');
        line.setAttribute('opacity', '0.7');
        svgElement.appendChild(line);

        // Add edge label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', (source.x + target.x) / 2);
        text.setAttribute('y', (source.y + target.y) / 2);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '10');
        text.setAttribute('fill', '#888');
        text.textContent = edge.type;
        svgElement.appendChild(text);
      }
    });

    // Draw nodes
    nodes.forEach(node => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', node.x);
      circle.setAttribute('cy', node.y);
      circle.setAttribute('r', getNodeRadius(node.type));
      circle.setAttribute('fill', getNodeColor(node.type));
      circle.setAttribute('stroke', '#333');
      circle.setAttribute('stroke-width', '2');
      svgElement.appendChild(circle);

      // Add node label
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', node.x);
      text.setAttribute('y', node.y + 25);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('font-size', '12');
      text.setAttribute('fill', '#e5e5e5');
      text.setAttribute('font-weight', 'bold');
      text.textContent = node.name.length > 10 ? node.name.substring(0, 10) + '...' : node.name;
      svgElement.appendChild(text);
    });

    svg.appendChild(svgElement);
  }, [graphData]);

  const getNodeColor = (type) => {
    switch (type) {
      case 'drug': return '#2f5288';
      case 'protein': return '#10b981';
      case 'disease': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getNodeRadius = (type) => {
    switch (type) {
      case 'drug': return 15;
      case 'protein': return 12;
      case 'disease': return 15;
      default: return 10;
    }
  };

  const getEdgeColor = (type) => {
    switch (type) {
      case 'inhibits': return '#ef4444';
      case 'activates': return '#10b981';
      case 'affects': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
    return (
      <div className="graph-placeholder">
        <p>No graph data available</p>
      </div>
    );
  }

  return (
    <div className="knowledge-graph-container">
      <h4>Knowledge Graph</h4>
      <div className="graph-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#2f5288' }}></div>
          <span>Drug</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#10b981' }}></div>
          <span>Protein</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: '#ef4444' }}></div>
          <span>Disease</span>
        </div>
      </div>
      <div ref={svgRef} className="graph-svg-container"></div>
    </div>
  );
};

export default KnowledgeGraphVisualization;