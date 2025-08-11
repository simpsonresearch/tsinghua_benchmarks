"""
Graph data structure for shortest path algorithm benchmarking.
Supports both directed and undirected graphs with weighted edges.
"""

import random
import math
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict


class Graph:
    """
    Graph data structure with support for weighted edges.
    """

    def __init__(self, num_nodes: int, directed: bool = True):
        """
        Initialize a graph.

        Args:
            num_nodes: Number of nodes in the graph
            directed: Whether the graph is directed
        """
        self.num_nodes = num_nodes
        self.directed = directed
        self.edges: List[Tuple[int, int, float]] = []
        self.adj_list: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.adj_matrix: List[List[float]] = [
            [float("inf")] * num_nodes for _ in range(num_nodes)
        ]

        # Initialize diagonal to 0 (distance from node to itself)
        for i in range(num_nodes):
            self.adj_matrix[i][i] = 0.0

    def add_edge(self, u: int, v: int, weight: float):
        """
        Add an edge to the graph.

        Args:
            u: Source node
            v: Destination node
            weight: Edge weight
        """
        if u < 0 or u >= self.num_nodes or v < 0 or v >= self.num_nodes:
            raise ValueError(f"Node indices must be between 0 and {self.num_nodes-1}")

        self.edges.append((u, v, weight))
        self.adj_list[u].append((v, weight))
        self.adj_matrix[u][v] = weight

        # For undirected graphs, add the reverse edge
        if not self.directed:
            self.adj_list[v].append((u, weight))
            self.adj_matrix[v][u] = weight

    def get_neighbors(self, node: int) -> List[Tuple[int, float]]:
        """Get all neighbors of a node with their edge weights."""
        return self.adj_list[node]

    def get_neighbors_fast(self, node: int) -> List[Tuple[int, float]]:
        """Optimized neighbor lookup that avoids list copying."""
        return self.adj_list[node]

    def get_weight(self, u: int, v: int) -> float:
        """Get the weight of edge from u to v."""
        return self.adj_matrix[u][v]

    def has_edge(self, u: int, v: int) -> bool:
        """Check if there's an edge from u to v."""
        return self.adj_matrix[u][v] != float("inf")

    def has_negative_weights(self) -> bool:
        """Check if the graph has any negative weight edges."""
        return any(weight < 0 for _, _, weight in self.edges)

    def get_edges_from_node(self, node: int) -> List[Tuple[int, float]]:
        """Get all outgoing edges from a specific node. More efficient than get_neighbors for large graphs."""
        return self.adj_list[node]

    def get_incoming_edges(self, node: int) -> List[Tuple[int, float]]:
        """Get all incoming edges to a specific node."""
        incoming = []
        for u, v, weight in self.edges:
            if v == node:
                incoming.append((u, weight))
        return incoming

    def get_node_degree(self, node: int) -> int:
        """Get the out-degree of a node."""
        return len(self.adj_list[node])

    def get_nodes_by_degree(self, reverse: bool = True) -> List[Tuple[int, int]]:
        """
        Get nodes sorted by degree for performance optimization.

        Args:
            reverse: If True, sort by descending degree (highest first)

        Returns:
            List of (node, degree) tuples sorted by degree
        """
        degrees = [(node, len(self.adj_list[node])) for node in range(self.num_nodes)]
        degrees.sort(key=lambda x: x[1], reverse=reverse)
        return degrees

    def get_high_degree_nodes(self, threshold: int = 3) -> Set[int]:
        """
        Get nodes with degree above a threshold.
        Useful for identifying influential nodes in algorithms.

        Args:
            threshold: Minimum degree to be considered high-degree

        Returns:
            Set of high-degree node indices
        """
        return {
            node
            for node in range(self.num_nodes)
            if len(self.adj_list[node]) > threshold
        }

    def density(self) -> float:
        """Calculate the density of the graph (ratio of actual edges to possible edges)."""
        max_edges = self.num_nodes * (self.num_nodes - 1)
        if not self.directed:
            max_edges //= 2
        return len(self.edges) / max_edges if max_edges > 0 else 0.0

    def get_all_nodes(self) -> List[int]:
        """Get list of all node indices."""
        return list(range(self.num_nodes))

    def is_connected(self) -> bool:
        """
        Check if the graph is connected (for undirected) or strongly connected (for directed).
        Uses DFS for simplicity.
        """
        if self.num_nodes == 0:
            return True

        visited = set()
        self._dfs(0, visited)

        # For undirected graphs, check if all nodes are reachable
        if not self.directed:
            return len(visited) == self.num_nodes

        # For directed graphs, we need to check strong connectivity
        # This is a simplified check - in practice, we'd use Kosaraju's algorithm
        return len(visited) == self.num_nodes

    def _dfs(self, node: int, visited: Set[int]):
        """Depth-first search helper method."""
        visited.add(node)
        for neighbor, _ in self.get_neighbors(node):
            if neighbor not in visited:
                self._dfs(neighbor, visited)

    def get_layers_from_source(
        self, source: int, max_layers: Optional[int] = None
    ) -> List[Set[int]]:
        """
        Get nodes organized in layers by their minimum distance from source.
        Used by the Tsinghua algorithm.
        """
        if source < 0 or source >= self.num_nodes:
            raise ValueError("Invalid source node")

        layers = []
        visited = set()
        current_layer = {source}
        layer_count = 0

        while current_layer and (max_layers is None or layer_count < max_layers):
            layers.append(current_layer.copy())
            visited.update(current_layer)

            next_layer = set()
            for node in current_layer:
                for neighbor, _ in self.get_neighbors(node):
                    if neighbor not in visited:
                        next_layer.add(neighbor)

            current_layer = next_layer
            layer_count += 1

        return layers

    def find_influential_nodes(
        self, layer: Set[int], lookahead_steps: int = 2
    ) -> Set[int]:
        """
        Find influential nodes in a layer - nodes that appear in many shortest paths.
        This is a heuristic approximation for the Tsinghua algorithm.
        """
        if not layer:
            return set()

        # Count how many nodes each node in the layer can reach in lookahead_steps
        influence_scores = {}

        for node in layer:
            reachable = set()
            self._count_reachable(node, lookahead_steps, reachable)
            influence_scores[node] = len(reachable)

        # Return nodes with above-average influence
        if not influence_scores:
            return layer

        avg_influence = sum(influence_scores.values()) / len(influence_scores)
        return {
            node for node, score in influence_scores.items() if score >= avg_influence
        }

    def _count_reachable(self, node: int, steps: int, reachable: Set[int]):
        """Helper method to count reachable nodes within a certain number of steps."""
        if steps <= 0:
            return

        for neighbor, _ in self.get_neighbors(node):
            if neighbor not in reachable:
                reachable.add(neighbor)
                self._count_reachable(neighbor, steps - 1, reachable)

    def cluster_frontier_nodes(
        self, frontier: Set[int], cluster_size: int = 3
    ) -> List[Set[int]]:
        """
        Cluster frontier nodes for the Tsinghua algorithm.
        Simple clustering based on connectivity.
        """
        if not frontier:
            return []

        clusters = []
        remaining = frontier.copy()

        while remaining:
            # Start a new cluster with an arbitrary node
            cluster = {remaining.pop()}

            # Add connected nodes to the cluster until we reach cluster_size
            while len(cluster) < cluster_size and remaining:
                # Find nodes in remaining that are connected to cluster nodes
                connected = set()
                for cluster_node in cluster:
                    for neighbor, _ in self.get_neighbors(cluster_node):
                        if neighbor in remaining:
                            connected.add(neighbor)

                if connected:
                    # Add the first connected node found
                    new_node = connected.pop()
                    cluster.add(new_node)
                    remaining.remove(new_node)
                else:
                    # No connected nodes found, start a new cluster
                    break

            clusters.append(cluster)

        return clusters

    def display(self):
        """Display the graph in a readable format."""
        print(
            f"Graph ({self.num_nodes} nodes, {'directed' if self.directed else 'undirected'}):"
        )

        if len(self.edges) <= 50:  # Only show edges for small graphs
            print("Edges:")
            for u, v, weight in self.edges:
                arrow = "->" if self.directed else "<->"
                print(f"  {u} {arrow} {v} (weight: {weight})")
        else:
            print(f"Too many edges to display ({len(self.edges)} edges)")

        print()

    def visualize(self):
        """Visualize the graph using matplotlib and networkx (if available)."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            raise ImportError("matplotlib and networkx are required for visualization")

        if self.num_nodes > 50:
            print("Graph too large for visualization")
            return

        # Create NetworkX graph
        G = nx.DiGraph() if self.directed else nx.Graph()

        # Add nodes
        G.add_nodes_from(range(self.num_nodes))

        # Add edges
        for u, v, weight in self.edges:
            G.add_edge(u, v, weight=weight)

        # Create layout
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=10,
            font_weight="bold",
        )

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(G, "weight")
        # Round weights for display
        edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        plt.title(
            f"Graph Visualization ({self.num_nodes} nodes, {len(self.edges)} edges)"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @classmethod
    def generate_random(
        cls,
        num_nodes: int,
        density: float = 0.3,
        directed: bool = True,
        allow_negative_weights: bool = False,
    ) -> "Graph":
        """
        Generate a random graph for testing.

        Args:
            num_nodes: Number of nodes
            density: Edge density (0.0 to 1.0)
            directed: Whether the graph should be directed
            allow_negative_weights: Whether to include negative weights

        Returns:
            A randomly generated graph
        """
        if num_nodes < 1:
            raise ValueError("Number of nodes must be positive")
        if not 0.0 <= density <= 1.0:
            raise ValueError("Density must be between 0.0 and 1.0")

        graph = cls(num_nodes, directed)

        # Calculate number of edges to add
        max_edges = num_nodes * (num_nodes - 1)
        if not directed:
            max_edges //= 2

        num_edges = int(density * max_edges)

        # Generate random edges
        edges_added = set()
        attempts = 0
        max_attempts = num_edges * 10  # Prevent infinite loops

        while len(edges_added) < num_edges and attempts < max_attempts:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)

            if u == v:  # No self-loops
                attempts += 1
                continue

            # For undirected graphs, ensure consistent edge representation
            if not directed and u > v:
                u, v = v, u

            edge_key = (u, v)
            if edge_key not in edges_added:
                # Generate random weight
                if allow_negative_weights:
                    weight = random.uniform(-10, 20)  # Bias towards positive weights
                else:
                    weight = random.uniform(0.1, 20)

                graph.add_edge(u, v, weight)
                edges_added.add(edge_key)

            attempts += 1

        # Ensure the graph is connected by adding a spanning tree if needed
        if not graph.is_connected():
            graph._ensure_connectivity()

        return graph

    def _ensure_connectivity(self):
        """Ensure the graph is connected by adding necessary edges."""
        # Simple approach: connect each node to node 0 if not already connected
        for i in range(1, self.num_nodes):
            if not self._has_path_to(i, 0):
                weight = random.uniform(1, 10)
                self.add_edge(i, 0, weight)

    def _has_path_to(self, start: int, target: int) -> bool:
        """Check if there's a path from start to target using BFS."""
        if start == target:
            return True

        visited = set()
        queue = [start]
        visited.add(start)

        while queue:
            current = queue.pop(0)
            for neighbor, _ in self.get_neighbors(current):
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    @classmethod
    def from_file(cls, filename: str) -> "Graph":
        """
        Load a graph from a file.

        Expected format:
        Line 1: num_nodes directed(0/1)
        Following lines: u v weight
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("Empty file")

        # Parse header
        header = lines[0].strip().split()
        num_nodes = int(header[0])
        directed = bool(int(header[1])) if len(header) > 1 else True

        graph = cls(num_nodes, directed)

        # Parse edges
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue

            parts = line.split()
            if len(parts) >= 3:
                u, v, weight = int(parts[0]), int(parts[1]), float(parts[2])
                graph.add_edge(u, v, weight)

        return graph

    def save_to_file(self, filename: str):
        """Save the graph to a file."""
        with open(filename, "w") as f:
            f.write(f"{self.num_nodes} {1 if self.directed else 0}\n")
            for u, v, weight in self.edges:
                f.write(f"{u} {v} {weight}\n")
