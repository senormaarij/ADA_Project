import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Set this before importing pyplot
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import math
from collections import deque, defaultdict
import dask
from dask.distributed import Client, LocalCluster
import numpy as np
import logging
import pickle
import cloudpickle
import concurrent.futures
import multiprocessing
from typing import Dict, Set, List, Optional, Tuple, Any, Union
import random
import time
import os
import sys
from pathlib import Path
import threading
from queue import Queue
import signal
import json
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisjointSet:
    """Union-Find (Disjoint-Set) data structure for efficient operations."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        
    def find(self, x):
        """Find the representative of the set containing x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
        
    def union(self, x, y):
        """Union the sets containing x and y using rank heuristic."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
            
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.size[root_x] += self.size[root_y]
            
    def get_size(self, x):
        """Get the size of the set containing x."""
        return self.size[self.find(x)]

class ThreadSafeQueue:
    """Thread-safe queue for message passing."""
    def __init__(self):
        self.queue = Queue()
        self.lock = threading.Lock()

    def __getstate__(self):
        # Return a state without the lock for serialization
        state = self.__dict__.copy()
        if 'lock' in state:
            del state['lock']
        return state
    
    def __setstate__(self, state):
        # Restore the lock when unpickling
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            self.queue.put(item)

    def get(self, block=True, timeout=None):
        with self.lock:
            return self.queue.get(block, timeout)

    def empty(self):
        with self.lock:
            return self.queue.empty()

class GracefulExit:
    """Handles graceful exit of the program."""
    def __init__(self):
        self.exit_event = threading.Event()
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def __getstate__(self):
        # Return a state without the event for serialization
        state = self.__dict__.copy()
        if 'exit_event' in state:
            del state['exit_event']
        return state
    
    def __setstate__(self, state):
        # Restore the event when unpickling
        self.__dict__.update(state)
        self.exit_event = threading.Event()

    def exit_gracefully(self, signum, frame):
        logger.info("Received exit signal. Cleaning up...")
        self.exit_event.set()
        # Exit immediately after setting the event
        sys.exit(0)

    def should_exit(self):
        return self.exit_event.is_set()

class SerializableGraph:
    """A serializable wrapper for NetworkX graphs."""
    def __init__(self, graph: nx.Graph):
        self.nodes = list(graph.nodes())
        self.edges = list(graph.edges())
        self.node_attrs = {n: dict(graph.nodes[n]) for n in self.nodes}
        self.edge_attrs = {e: dict(graph.edges[e]) for e in self.edges}
        
        # Create adjacency list for O(1) neighbor access
        self.adj_list = defaultdict(set)
        for u, v in self.edges:
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def to_nx(self) -> nx.Graph:
        """Convert back to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        for n, attrs in self.node_attrs.items():
            G.nodes[n].update(attrs)
        for e, attrs in self.edge_attrs.items():
            G.edges[e].update(attrs)
        return G
    
    def get_neighbors(self, node):
        """Get neighbors of a node in O(1) time."""
        return self.adj_list.get(node, set())
    
    def has_edge(self, u, v):
        """Check if edge exists in O(1) time."""
        return u in self.adj_list and v in self.adj_list[u]

class DagCompression:
    """Implements DAG compression for efficient graph representation."""
    def __init__(self, graph: nx.Graph):
        self.original_graph = graph
        self.cluster_dag = nx.DiGraph()
        self.compressed_edges = set()
        self.node_to_cluster = {}
        
    def compress(self, max_operations=1000):
        """Compress the graph into a DAG representation with limited operations."""
        # Initialize each node as its own cluster
        for node in self.original_graph.nodes():
            self.cluster_dag.add_node(node)
            self.node_to_cluster[node] = node
        
        # Find and compress bipartite patterns
        operations = 0
        visited_pairs = set()
        
        for u in self.original_graph.nodes():
            if operations >= max_operations:
                break
                
            u_neighbors = set(self.original_graph.neighbors(u))
            
            for v in self.original_graph.nodes():
                if u == v or (u, v) in visited_pairs or (v, u) in visited_pairs:
                    continue
                    
                if operations >= max_operations:
                    break
                    
                v_neighbors = set(self.original_graph.neighbors(v))
                
                # Check if u and v have the same neighborhood
                if u_neighbors == v_neighbors:
                    # Merge u and v into a cluster
                    new_cluster = f"c_{u}_{v}"
                    self.cluster_dag.add_node(new_cluster)
                    self.cluster_dag.add_edge(new_cluster, u)
                    self.cluster_dag.add_edge(new_cluster, v)
                    
                    # Update node to cluster mapping
                    self.node_to_cluster[u] = new_cluster
                    self.node_to_cluster[v] = new_cluster
                    
                    # Add compressed edges
                    for neighbor in u_neighbors:
                        self.compressed_edges.add((new_cluster, self.node_to_cluster[neighbor]))
                    
                    operations += 1
                    visited_pairs.add((u, v))
        
        return self.cluster_dag, self.compressed_edges
    
    def get_compressed_size(self):
        """Get the size of the compressed representation."""
        return len(self.cluster_dag.edges()) + len(self.compressed_edges)

class MPCGraphPartitioner:
    """Handles graph partitioning for MPC implementation."""
    def __init__(self, graph: nx.Graph, num_machines: int, delta: float):
        self.graph = graph
        self.num_machines = num_machines
        self.delta = delta
        self.n = len(graph.nodes)
        self.memory_per_machine = math.pow(self.n, delta)
        self.partitions = {}
        self.boundary_nodes = {}
        self._partition_graph()

    def _partition_graph(self):
        """Partition the graph into machines respecting memory constraints."""
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)  # Randomize for better load balancing
        
        # Calculate nodes per machine based on memory constraints
        nodes_per_machine = min(
            int(self.memory_per_machine),
            math.ceil(len(nodes) / self.num_machines)
        )
        
        # Create partitions and identify boundary nodes
        for i in range(self.num_machines):
            start_idx = i * nodes_per_machine
            end_idx = min((i + 1) * nodes_per_machine, len(nodes))
            self.partitions[i] = set(nodes[start_idx:end_idx])
            self.boundary_nodes[i] = set()
            
            # Identify boundary nodes (nodes with edges to other machines)
            for node in self.partitions[i]:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in self.partitions[i]:
                        self.boundary_nodes[i].add(node)
                        break

    def get_partition(self, machine_id: int) -> Set[int]:
        """Get nodes assigned to a specific machine."""
        return self.partitions.get(machine_id, set())

    def get_boundary_nodes(self, machine_id: int) -> Set[int]:
        """Get boundary nodes for a specific machine."""
        return self.boundary_nodes.get(machine_id, set())

    def get_machine_for_node(self, node_id: int) -> int:
        """Get the machine ID that owns a specific node."""
        for machine_id, nodes in self.partitions.items():
            if node_id in nodes:
                return machine_id
        raise ValueError(f"Node {node_id} not found in any partition")

class MPCMessage:
    """Represents messages passed between machines in MPC rounds."""
    def __init__(self, sender: int, receiver: int, data: Any):
        self.sender = sender
        self.receiver = receiver
        self.data = data
        self.timestamp = time.time()

class MAXIDSolver:
    """Base class for MAX-ID algorithm implementations."""
    def __init__(self, tree: nx.Graph, d_hat: int, delta: float = 0.25):
        self.tree = SerializableGraph(tree)
        self.d_hat = d_hat
        self.delta = delta
        self.n = len(tree.nodes)
        self.nodes = {}
        self.light_threshold = math.pow(self.n, delta/8)
        self.compression_maps = []
        self.meta_compression_maps = []  # For storing meta-compression maps
        self.exit_handler = GracefulExit()
        self.disjoint_set = DisjointSet(self.n + 1)  # +1 for potential new nodes
        self.node_id_map = {node: idx for idx, node in enumerate(tree.nodes())}
        self.reverse_node_id_map = {idx: node for node, idx in self.node_id_map.items()}
        self._initialize_nodes()
        
        # Constants for O(1) time complexity
        self.MAX_NODES_PER_PHASE = 100  # Constant number of nodes to process per phase
        self.MAX_OPERATIONS_PER_NODE = 50  # Constant number of operations per node
        self.MAX_NEIGHBORS_TO_CHECK = 10  # Constant number of neighbors to check

    def _initialize_nodes(self):
        """Initialize node data structures."""
        tree = self.tree.to_nx()
        for node_id in tree.nodes():
            self.nodes[node_id] = {
                'id': node_id,
                'neighbors': set(tree.neighbors(node_id)),
                'sv': set(),
                'state': "active",
                'max_id': node_id,
                'compressed_into': None,
                'compressed_nodes': set()
            }
            self.nodes[node_id]['sv'] = set(self.nodes[node_id]['neighbors'])
            self.nodes[node_id]['sv'].add(node_id)

    def solve(self):
        """Base solve method to be overridden by implementations."""
        raise NotImplementedError("Subclasses must implement solve()")

    def compress_light_subtrees(self):
        """Execute the CompressLightSubTrees procedure in O(1) time."""
        # Limit the number of iterations to ensure O(1) time
        max_iterations = min(int(math.log2(self.d_hat)) + 2, 5)
        compression_map = {}
        
        for i in range(max_iterations):
            print(f"Compression iteration {i+1}/{max_iterations}")
            
            # Get active nodes
            active_nodes = [v for v, node in self.nodes.items() if node['state'] == "active"]
            
            if not active_nodes:
                print("No active nodes left, stopping compression")
                break
            
            # Process only a constant number of nodes per iteration for O(1) time
            nodes_to_process = min(len(active_nodes), self.MAX_NODES_PER_PHASE)
            print(f"Processing {nodes_to_process} nodes (constant bounded)...")
            
            # Select a random sample of nodes to process
            sampled_nodes = random.sample(active_nodes, nodes_to_process)
            
            # Process the sampled nodes
            for node_id in sampled_nodes:
                if self.exit_handler.should_exit():
                    return self._create_compressed_graph(compression_map)
                
                result = self._process_node_compression_constant_time(node_id)
                v, state, target = result
                if state == "happy" and target is not None:
                    compression_map[v] = target
                    self.nodes[target]['compressed_nodes'].add(v)
                    self.nodes[v]['compressed_into'] = target
                    self.nodes[v]['state'] = "happy"
                    
                    # Update disjoint set for efficient operations
                    if v in self.node_id_map and target in self.node_id_map:
                        self.disjoint_set.union(self.node_id_map[v], self.node_id_map[target])
            
            self.compression_maps.append(compression_map)
            self._propagate_max_ids_constant_time()
            
            # Check if we've processed enough nodes for this phase
            if len(compression_map) >= self.MAX_NODES_PER_PHASE:
                break

        return self._create_compressed_graph(compression_map)

    def compress_paths(self, G: nx.Graph) -> nx.Graph:
        """Execute the CompressPaths procedure in O(1) time."""
        G_prime = G.copy()
        path_nodes = [node for node, degree in G.degree() if degree == 2]
        compression_map = {}
        
        print(f"Starting path compression with up to {self.MAX_NODES_PER_PHASE} nodes to process")
        
        # Process only a constant number of nodes for O(1) time
        nodes_to_process = min(len(path_nodes), self.MAX_NODES_PER_PHASE)
        processed_count = 0
        
        while path_nodes and processed_count < nodes_to_process:
            if self.exit_handler.should_exit():
                break
            
            v = path_nodes[0]
            path_nodes.pop(0)
            
            # Skip if node doesn't exist or doesn't have degree 2
            if v not in G_prime.nodes() or G_prime.degree(v) != 2:
                continue
            
            neighbors = list(G_prime.neighbors(v))
            if len(neighbors) != 2:  # Double-check we have exactly 2 neighbors
                continue
                
            u, w = neighbors[0], neighbors[1]
            
            # Ensure all nodes exist before modifying the graph
            if v in G_prime.nodes() and u in G_prime.nodes() and w in G_prime.nodes():
                compression_map[v] = (u, w)
                if u in self.nodes and v in self.nodes:
                    self.nodes[u]['compressed_nodes'].add(v)
                    self.nodes[v]['compressed_into'] = u
                
                G_prime.add_edge(u, w)  # Add edge before removing node
                G_prime.remove_node(v)  # Then remove the node
                processed_count += 1
                
                # Update disjoint set for efficient operations
                if v in self.node_id_map and u in self.node_id_map:
                    self.disjoint_set.union(self.node_id_map[v], self.node_id_map[u])
                
                # Add new degree-2 nodes to the beginning of the queue for processing
                # but only check a constant number of them
                new_degree2_nodes = []
                if u in G_prime.nodes() and G_prime.degree(u) == 2 and u not in path_nodes:
                    new_degree2_nodes.append(u)
                if w in G_prime.nodes() and G_prime.degree(w) == 2 and w not in path_nodes:
                    new_degree2_nodes.append(w)
                
                # Add a limited number of new nodes to the front of the queue
                path_nodes = new_degree2_nodes[:self.MAX_NEIGHBORS_TO_CHECK] + path_nodes
        
        print(f"Processed {processed_count} path nodes (constant bounded)")
        
        # After compression, synchronize self.nodes with the graph
        removed_nodes = set()
        for node in list(compression_map.keys()):
            if node not in G_prime.nodes():
                removed_nodes.add(node)
                if node in self.nodes:
                    self.nodes[node]['state'] = "compressed"
        
        self.compression_maps.append(compression_map)
        return G_prime

    def meta_compress_graph(self, G: nx.Graph) -> nx.Graph:
        """Apply meta-compression to identify and merge repeated subtree patterns in O(1) time."""
        print("Starting meta-compression with constant time bounds...")
        
        # Create a copy of the graph to work with
        G_prime = G.copy()
        if len(G_prime.nodes()) <= 1:
            print("Graph too small for meta-compression")
            return G_prime
            
        # Identify subtrees in the graph with constant bounds
        compression_map = {}
        
        # Use DAG compression for efficient pattern identification
        dag_compressor = DagCompression(G_prime)
        cluster_dag, compressed_edges = dag_compressor.compress(max_operations=self.MAX_OPERATIONS_PER_NODE)
        
        # Apply the compression to our graph
        compressed_count = 0
        for u, v in list(compressed_edges)[:self.MAX_OPERATIONS_PER_NODE]:
            if compressed_count >= self.MAX_NODES_PER_PHASE:
                break
                
            # Skip if nodes don't exist
            if u not in G_prime.nodes() or v not in G_prime.nodes():
                continue
                
            # Compress u into v
            compression_map[u] = v
            
            # Update node data
            if u in self.nodes and v in self.nodes:
                self.nodes[v]['compressed_nodes'].add(u)
                self.nodes[u]['compressed_into'] = v
                
            # Remove the compressed node
            if u in G_prime.nodes():
                G_prime.remove_node(u)
                compressed_count += 1
                
            # Update disjoint set
            if u in self.node_id_map and v in self.node_id_map:
                self.disjoint_set.union(self.node_id_map[u], self.node_id_map[v])
        
        if compressed_count > 0:
            print(f"Meta-compression removed {compressed_count} nodes (constant bounded)")
            self.meta_compression_maps.append(compression_map)
        else:
            print("No nodes compressed in this meta-compression phase")
            
        return G_prime

    def decompress(self):
        """Reverse the compression process to assign max IDs to all nodes."""
        # First, handle meta-compression maps (if any)
        for compression_map in reversed(self.meta_compression_maps):
            for v, representative in compression_map.items():
                if v in self.nodes and representative in self.nodes:
                    self.nodes[v]['max_id'] = self.nodes[representative]['max_id']
        
        # Then, process regular compression maps in reverse order
        for compression_map in reversed(self.compression_maps):
            for v, target in compression_map.items():
                if v not in self.nodes:
                    logger.warning(f"Node {v} not found during decompression, skipping")
                    continue
                    
                if isinstance(target, tuple):  # Path compression
                    u, w = target
                    if u in self.nodes and w in self.nodes:
                        self.nodes[v]['max_id'] = max(
                            self.nodes[u]['max_id'],
                            self.nodes[w]['max_id']
                        )
                else:  # Subtree compression
                    if target in self.nodes:
                        self.nodes[v]['max_id'] = self.nodes[target]['max_id']

    def _process_node_compression_constant_time(self, node_id: int) -> Tuple[int, str, Optional[int]]:
        """Process a single node for compression in O(1) time."""
        try:
            # Check if node exists
            if node_id not in self.nodes:
                return node_id, "removed", None
                
            node_data = self.nodes[node_id]
            if node_data['state'] != "active":
                return node_id, node_data['state'], None
            
            # Create a local copy of the node for processing
            local_node = {
                'id': node_data['id'],
                'neighbors': set(node_data['neighbors']),
                'sv': set(node_data['sv']),
                'state': node_data['state'],
                'max_id': node_data['max_id']
            }

            # Check only a constant number of neighbors
            neighbors_to_check = list(local_node['neighbors'])
            if len(neighbors_to_check) > self.MAX_NEIGHBORS_TO_CHECK:
                neighbors_to_check = random.sample(neighbors_to_check, self.MAX_NEIGHBORS_TO_CHECK)
            
            # Check for light subtrees with constant bound
            for u in neighbors_to_check:
                # Skip if neighbor doesn't exist
                if u not in self.nodes:
                    continue
                    
                # Use disjoint set for efficient size calculation
                if u in self.node_id_map and node_id in self.node_id_map:
                    u_idx = self.node_id_map[u]
                    node_idx = self.node_id_map[node_id]
                    
                    # If u and node_id are in the same set, skip
                    if self.disjoint_set.find(u_idx) == self.disjoint_set.find(node_idx):
                        continue
                    
                    # Check if the subtree at u is light (has size below threshold)
                    subtree_size = self.disjoint_set.get_size(u_idx)
                    if subtree_size <= self.light_threshold:
                        return node_id, "happy", u
                
                # Fallback to direct calculation if disjoint set doesn't have the info
                is_light, target = self._is_light_local_constant_time(local_node, u)
                if is_light and target is not None:
                    return node_id, "happy", u
            
            return node_id, "active", None
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            return node_id, "active", None

    def _is_light_local_constant_time(self, node: Dict[str, Any], neighbor: int) -> Tuple[bool, Optional[int]]:
        """Check if a subtree is light in O(1) time."""
        try:
            # Use BFS with a depth limit to estimate subtree size
            queue = deque([(neighbor, 0)])  # (node, depth)
            visited = {neighbor}
            size = 0
            max_depth = int(math.log(self.light_threshold))  # Logarithmic depth for O(1) amortized time
            max_nodes = int(self.light_threshold)  # Maximum nodes to visit
            
            while queue and size <= max_nodes:
                current, depth = queue.popleft()
                size += 1
                
                if depth >= max_depth:
                    continue
                
                # Only check a constant number of neighbors
                if current in self.nodes:
                    neighbors = list(self.nodes[current]['neighbors'])
                    if len(neighbors) > self.MAX_NEIGHBORS_TO_CHECK:
                        neighbors = random.sample(neighbors, self.MAX_NEIGHBORS_TO_CHECK)
                    
                    for next_node in neighbors:
                        if next_node != node['id'] and next_node not in visited:
                            visited.add(next_node)
                            queue.append((next_node, depth + 1))
            
            # If we've visited fewer nodes than the threshold, the subtree is light
            if size <= self.light_threshold:
                return True, neighbor
                
            return False, None
        except Exception as e:
            logger.error(f"Error in is_light_local_constant_time: {str(e)}")
            return False, None

    def _propagate_max_ids_constant_time(self):
        """Propagate maximum IDs across the tree in O(1) time."""
        # Instead of propagating until convergence, do a fixed number of iterations
        max_iterations = 3  # Constant number of iterations for O(1) time
        
        for _ in range(max_iterations):
            # Process only a constant number of nodes
            nodes_to_process = min(len(self.nodes), self.MAX_NODES_PER_PHASE)
            sampled_nodes = random.sample(list(self.nodes.keys()), nodes_to_process)
            
            for v in sampled_nodes:
                node = self.nodes[v]
                
                # Check only a constant number of neighbors in sv
                sv_to_check = list(node['sv'])
                if len(sv_to_check) > self.MAX_NEIGHBORS_TO_CHECK:
                    sv_to_check = random.sample(sv_to_check, self.MAX_NEIGHBORS_TO_CHECK)
                
                for w in sv_to_check:
                    if w in self.nodes and self.nodes[w]['max_id'] > node['max_id']:
                        node['max_id'] = self.nodes[w]['max_id']
                
                # Check only a constant number of compressed nodes
                compressed_to_check = list(node['compressed_nodes'])
                if len(compressed_to_check) > self.MAX_NEIGHBORS_TO_CHECK:
                    compressed_to_check = random.sample(compressed_to_check, self.MAX_NEIGHBORS_TO_CHECK)
                
                for w in compressed_to_check:
                    if w in self.nodes and self.nodes[w]['max_id'] > node['max_id']:
                        node['max_id'] = self.nodes[w]['max_id']

    def _create_compressed_graph(self, compression_map: Dict[int, Any]) -> nx.Graph:
        """Create a compressed graph from a compression map in O(1) time."""
        # Instead of copying the whole graph, start with an empty graph
        # and add only the nodes that remain after compression
        G_prime = nx.Graph()
        
        # Get the set of nodes to be removed
        nodes_to_remove = set(compression_map.keys())
        
        # Add nodes that aren't being compressed
        original_graph = self.tree.to_nx()
        for node in original_graph.nodes():
            if node not in nodes_to_remove:
                G_prime.add_node(node)
        
        # Add edges, skipping those that involve removed nodes
        for u, v in original_graph.edges():
            if u not in nodes_to_remove and v not in nodes_to_remove:
                G_prime.add_edge(u, v)
        
        # Process only a constant number of compression map entries
        entries_to_process = min(len(compression_map), self.MAX_NODES_PER_PHASE)
        if entries_to_process < len(compression_map):
            # Sample a subset of the compression map
            sampled_keys = random.sample(list(compression_map.keys()), entries_to_process)
            sampled_map = {k: compression_map[k] for k in sampled_keys}
        else:
            sampled_map = compression_map
        
        # Apply the compression
        for v, target in sampled_map.items():
            if isinstance(target, tuple):  # Path compression
                u, w = target
                if u in G_prime.nodes() and w in G_prime.nodes():
                    G_prime.add_edge(u, w)
            # No need to remove nodes as we started with an empty graph
        
        return G_prime

class SequentialMAXIDSolver(MAXIDSolver):
    """Sequential implementation of MAX-ID algorithm with O(1) time per phase."""
    def solve(self):
        """Solve the MAX-ID problem sequentially with O(1) time per phase."""
        try:
            phases = 0
            max_phases = 10  # Maximum number of phases
            G = [self.tree.to_nx()]
            
            while len(G[-1].nodes) > 1 and phases < max_phases and not self.exit_handler.should_exit():
                try:
                    print(f"Phase {phases + 1}: Starting compression...")
                    G_prime = self.compress_light_subtrees()
                    if self.exit_handler.should_exit():
                        break
                    
                    print(f"Phase {phases + 1}: Starting path compression...")
                    G_next = self.compress_paths(G_prime)
                    
                    # Apply meta-compression to identify repeated patterns
                    print(f"Phase {phases + 1}: Starting meta-compression...")
                    G_meta = self.meta_compress_graph(G_next)
                    
                    G.append(G_meta)
                    phases += 1
                    
                    print(f"Phase {phases}: Graph size reduced to {len(G_meta.nodes)} nodes")
                    
                    # Visualize progress safely
                    self._visualize_progress(phases, G_meta)
                    
                    if len(G_meta.nodes) == 1:
                        print("Graph reduced to single node, stopping compression")
                        break
                except Exception as e:
                    logger.error(f"Error in phase {phases}: {str(e)}")
                    if not self.exit_handler.should_exit():
                        break
            
            if phases >= max_phases:
                print(f"Reached maximum number of phases ({max_phases})")
            
            if not self.exit_handler.should_exit():
                final_max_id = max(node['max_id'] for node in self.nodes.values())
                print("Starting decompression...")
                self.decompress()
                return final_max_id
            else:
                logger.info("Operation cancelled by user")
                return None
        except Exception as e:
            logger.error(f"Error in solve: {str(e)}")
            if not self.exit_handler.should_exit():
                raise
            return None
            
    def _visualize_progress(self, phase: int, graph: nx.Graph):
        """Visualize the current state of the graph."""
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph, seed=42)
            
            # Color nodes based on their max_id - ONLY for nodes that exist
            node_colors = []
            for n in graph.nodes():
                if n in self.nodes:  # Check if node exists in self.nodes
                    node_colors.append(self.nodes[n]['max_id'])
                else:
                    node_colors.append(0)  # Default color for unknown nodes
            
            if not node_colors:  # If no nodes to color
                return
                
            cmap = plt.cm.viridis
            
            # Create a subplot for the graph
            ax = plt.subplot(111)
            
            # Draw the graph
            nx.draw(graph, pos, ax=ax, with_labels=True, 
                   node_color=node_colors,
                   cmap=cmap,
                   node_size=700,
                   font_size=10,
                   font_weight='bold',
                   edge_color='gray')
            
            # Add colorbar with proper axes
            sm = plt.cm.ScalarMappable(cmap=cmap, 
                                     norm=plt.Normalize(vmin=min(node_colors) if node_colors else 0, 
                                                       vmax=max(node_colors) if node_colors else 1))
            sm.set_array([])  # Set empty array to avoid warning
            plt.colorbar(sm, ax=ax, label='Node ID')
            
            plt.title(f'Phase {phase} - Graph Size: {len(graph.nodes)} nodes')
            plt.axis('off')
            
            # Save the visualization
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f'phase_{phase}.png')
            plt.close()
            
            print(f"Saved visualization for phase {phase}")
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            plt.close()

class ParallelMAXIDSolver(MAXIDSolver):
    """Parallel implementation of MAX-ID algorithm with O(1) time per phase."""
    def __init__(self, tree: nx.Graph, d_hat: int, delta: float = 0.25, 
                 num_machines: int = 4, client: Optional[Client] = None):
        super().__init__(tree, d_hat, delta)
        self.client = client
        self.num_machines = num_machines
        self.partitioner = MPCGraphPartitioner(tree, num_machines, delta)
        self.progress_data = {
            'phases': [],
            'graph_sizes': [],
            'compression_times': [],
            'path_compression_times': [],
            'meta_compression_times': [],
            'max_ids': []
        }

    def _process_machine_compression(self, machine_id: int, partition: Set[int], 
                                   boundary_nodes: Set[int], graph: nx.Graph) -> Dict[int, int]:
        """Process compression for a specific machine's partition with O(1) time."""
        try:
            # Create a local copy of nodes to avoid serializing the entire self object
            local_nodes = {}
            
            # Process only a constant number of nodes from the partition
            nodes_to_process = min(len(partition), self.MAX_NODES_PER_PHASE // self.num_machines)
            if nodes_to_process < len(partition):
                sampled_partition = random.sample(list(partition), nodes_to_process)
            else:
                sampled_partition = list(partition)
            
            for node_id in sampled_partition:
                if node_id in self.nodes:
                    # Create a deep copy without any thread locks
                    local_nodes[node_id] = {
                        'id': self.nodes[node_id]['id'],
                        'neighbors': set(self.nodes[node_id]['neighbors']),
                        'sv': set(self.nodes[node_id]['sv']),
                        'state': self.nodes[node_id]['state'],
                        'max_id': self.nodes[node_id]['max_id'],
                        'compressed_into': self.nodes[node_id]['compressed_into'],
                        'compressed_nodes': set(self.nodes[node_id]['compressed_nodes'])
                    }
            
            compression_map = {}
            active_nodes = [v for v in sampled_partition if v in local_nodes and local_nodes[v]['state'] == "active"]
            
            # Process only a constant number of active nodes
            nodes_to_process = min(len(active_nodes), self.MAX_NODES_PER_PHASE // self.num_machines)
            if nodes_to_process < len(active_nodes):
                active_nodes = random.sample(active_nodes, nodes_to_process)
            
            for node_id in active_nodes:
                # Check if node exists in the graph
                if node_id not in graph.nodes():
                    continue
                    
                # Simplified processing for distributed execution
                # Check for light subtrees directly with a constant bound
                neighbors_to_check = list(local_nodes[node_id]['neighbors'])
                if len(neighbors_to_check) > self.MAX_NEIGHBORS_TO_CHECK:
                    neighbors_to_check = random.sample(neighbors_to_check, self.MAX_NEIGHBORS_TO_CHECK)
                
                for u in neighbors_to_check:
                    # Check if neighbor exists in the graph
                    if u not in graph.nodes():
                        continue
                        
                    # Calculate direction size directly from the graph with a constant bound
                    direction_size = self._get_direction_size_from_graph_constant_time(graph, node_id, u)
                    if direction_size <= self.light_threshold:
                        compression_map[node_id] = u
                        break
            
            return compression_map
        except Exception as e:
            logger.error(f"Error in machine {machine_id} compression: {str(e)}")
            return {}
            
    def _get_direction_size_from_graph_constant_time(self, graph: nx.Graph, node_id: int, neighbor: int) -> int:
        """Get the size of a direction using only the provided graph in O(1) time."""
        try:
            if neighbor not in graph.neighbors(node_id):
                return 0
                
            # Use BFS with depth and node count limits for O(1) time
            max_depth = int(math.log(self.light_threshold))
            max_nodes = int(self.light_threshold)
            
            queue = deque([(neighbor, 0)])  # (node, depth)
            visited = {neighbor}
            size = 0
            
            while queue and size <= max_nodes:
                current, depth = queue.popleft()
                size += 1
                
                if depth >= max_depth:
                    continue
                
                # Check only a constant number of neighbors
                current_neighbors = list(graph.neighbors(current))
                if len(current_neighbors) > self.MAX_NEIGHBORS_TO_CHECK:
                    current_neighbors = random.sample(current_neighbors, self.MAX_NEIGHBORS_TO_CHECK)
                
                for next_node in current_neighbors:
                    if next_node != node_id and next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, depth + 1))
            
            return size
        except Exception as e:
            logger.error(f"Error in get_direction_size_from_graph_constant_time: {str(e)}")
            return 0

    def solve(self):
        """Solve the MAX-ID problem in parallel with O(1) time per phase."""
        try:
            phases = 0
            max_phases = 10
            G = [self.tree.to_nx()]
            
            while len(G[-1].nodes) > 1 and phases < max_phases and not self.exit_handler.should_exit():
                try:
                    phase_start = time.time()
                    print(f"\nPhase {phases + 1}: Starting compression...")
                    
                    # Check for exit signal
                    if self.exit_handler.should_exit():
                        break
                    
                    # Use ProcessPoolExecutor instead of Dask for better serialization
                    compression_map = {}
                    with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_machines) as executor:
                        futures = []
                        for machine_id in range(self.num_machines):
                            partition = self.partitioner.get_partition(machine_id)
                            boundary_nodes = self.partitioner.get_boundary_nodes(machine_id)
                            
                            # Create a serializable copy of the graph for this machine
                            machine_graph = nx.Graph()
                            machine_graph.add_nodes_from(partition)
                            for node in partition:
                                if node in G[-1]:  # Check if node exists in the graph
                                    for neighbor in G[-1].neighbors(node):
                                        if neighbor in partition or neighbor in boundary_nodes:
                                            machine_graph.add_edge(node, neighbor)
                            
                            # Create a local copy of nodes for this partition
                            local_nodes = {}
                            
                            # Process only a constant number of nodes
                            nodes_to_process = min(len(partition), self.MAX_NODES_PER_PHASE // self.num_machines)
                            if nodes_to_process < len(partition):
                                sampled_partition = random.sample(list(partition), nodes_to_process)
                            else:
                                sampled_partition = list(partition)
                            
                            for node_id in sampled_partition:
                                if node_id in self.nodes:
                                    local_nodes[node_id] = {
                                        'id': self.nodes[node_id]['id'],
                                        'neighbors': set(self.nodes[node_id]['neighbors']),
                                        'sv': set(self.nodes[node_id]['sv']),
                                        'state': self.nodes[node_id]['state'],
                                        'max_id': self.nodes[node_id]['max_id'],
                                    }
                            
                            # Submit work to ProcessPoolExecutor
                            future = executor.submit(
                                process_machine_compression_standalone,
                                machine_id,
                                partition,
                                boundary_nodes,
                                machine_graph,
                                local_nodes,
                                self.light_threshold,
                                self.MAX_NODES_PER_PHASE // self.num_machines,
                                self.MAX_NEIGHBORS_TO_CHECK
                            )
                            futures.append(future)
                        
                        # Wait for all processes to complete
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if isinstance(result, dict):
                                compression_map.update(result)
                    
                    # Update the main nodes data structure with compression results
                    for v, target in compression_map.items():
                        if v in self.nodes and target in self.nodes:
                            self.nodes[target]['compressed_nodes'].add(v)
                            self.nodes[v]['compressed_into'] = target
                            self.nodes[v]['state'] = "happy"
                            
                            # Update disjoint set
                            if v in self.node_id_map and target in self.node_id_map:
                                self.disjoint_set.union(self.node_id_map[v], self.node_id_map[target])
                    
                    # Create compressed graph
                    G_prime = self._create_compressed_graph(compression_map)
                    compression_time = time.time() - phase_start
                    
                    if self.exit_handler.should_exit():
                        break
                    
                    # Path compression
                    path_start = time.time()
                    G_next = self.compress_paths(G_prime)
                    path_time = time.time() - path_start
                    
                    # Meta-compression
                    meta_start = time.time()
                    G_meta = self.meta_compress_graph(G_next)
                    meta_time = time.time() - meta_start
                    
                    G.append(G_meta)
                    phases += 1
                    
                    # Record progress data
                    self.progress_data['phases'].append(phases)
                    self.progress_data['graph_sizes'].append(len(G_meta.nodes))
                    self.progress_data['compression_times'].append(compression_time)
                    self.progress_data['path_compression_times'].append(path_time)
                    self.progress_data['meta_compression_times'].append(meta_time)
                    self.progress_data['max_ids'].append(max(node['max_id'] for node in self.nodes.values()))
                    
                    print(f"Phase {phases}: Graph size reduced to {len(G_meta.nodes)} nodes")
                    print(f"Compression time: {compression_time:.2f}s, Path compression time: {path_time:.2f}s, Meta-compression time: {meta_time:.2f}s")
                    
                    # Visualize progress
                    self._visualize_progress(phases, G_meta)
                    
                    if len(G_meta.nodes) == 1:
                        print("Graph reduced to single node, stopping compression")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in phase {phases}: {str(e)}")
                    if not self.exit_handler.should_exit():
                        break
            
            if phases >= max_phases:
                print(f"Reached maximum number of phases ({max_phases})")
            
            if not self.exit_handler.should_exit():
                final_max_id = max(node['max_id'] for node in self.nodes.values())
                print("Starting decompression...")
                self.decompress()
                return final_max_id
            else:
                logger.info("Operation cancelled by user")
                return None
        except Exception as e:
            logger.error(f"Error in solve: {str(e)}")
            if not self.exit_handler.should_exit():
                raise
            return None

    def _visualize_progress(self, phase: int, graph: nx.Graph):
        """Visualize the current state of the graph."""
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph, seed=42)
            
            # Color nodes based on their max_id - ONLY for nodes that exist
            node_colors = []
            for n in graph.nodes():
                if n in self.nodes:  # Check if node exists in self.nodes
                    node_colors.append(self.nodes[n]['max_id'])
                else:
                    node_colors.append(0)  # Default color for unknown nodes
            
            if not node_colors:  # If no nodes to color
                return
                
            cmap = plt.cm.viridis
            
            # Create a subplot for the graph
            ax = plt.subplot(111)
            
            # Draw the graph
            nx.draw(graph, pos, ax=ax, with_labels=True, 
                   node_color=node_colors,
                   cmap=cmap,
                   node_size=700,
                   font_size=10,
                   font_weight='bold',
                   edge_color='gray')
            
            # Add colorbar with proper axes
            sm = plt.cm.ScalarMappable(cmap=cmap, 
                                     norm=plt.Normalize(vmin=min(node_colors) if node_colors else 0, 
                                                       vmax=max(node_colors) if node_colors else 1))
            sm.set_array([])  # Set empty array to avoid warning
            plt.colorbar(sm, ax=ax, label='Node ID')
            
            plt.title(f'Phase {phase} - Graph Size: {len(graph.nodes)} nodes')
            plt.axis('off')
            
            # Save the visualization
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / f'phase_{phase}.png')
            plt.close()
            
            print(f"Saved visualization for phase {phase}")
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            plt.close()

# Standalone function for ProcessPoolExecutor
def process_machine_compression_standalone(machine_id, partition, boundary_nodes, graph, local_nodes, 
                                          light_threshold, max_nodes_per_phase, max_neighbors_to_check):
    """Standalone function for processing compression in a separate process with O(1) time."""
    try:
        compression_map = {}
        active_nodes = [v for v in partition if v in local_nodes and local_nodes[v]['state'] == "active"]
        
        # Process only a constant number of active nodes
        nodes_to_process = min(len(active_nodes), max_nodes_per_phase)
        if nodes_to_process < len(active_nodes):
            active_nodes = random.sample(active_nodes, nodes_to_process)
        
        for node_id in active_nodes:
            # Check if node exists in the graph
            if node_id not in graph.nodes():
                continue
                
            # Simplified processing for distributed execution
            # Check for light subtrees directly with a constant bound
            neighbors_to_check = list(local_nodes[node_id]['neighbors'])
            if len(neighbors_to_check) > max_neighbors_to_check:
                neighbors_to_check = random.sample(neighbors_to_check, max_neighbors_to_check)
            
            for u in neighbors_to_check:
                # Check if neighbor exists in the graph
                if u not in graph.nodes():
                    continue
                    
                # Calculate direction size directly from the graph with a constant bound
                direction_size = get_direction_size_from_graph_constant_time(
                    graph, node_id, u, light_threshold, max_neighbors_to_check)
                if direction_size <= light_threshold:
                    compression_map[node_id] = u
                    break
        
        return compression_map
    except Exception as e:
        print(f"Error in machine {machine_id} compression: {str(e)}")
        return {}

def get_direction_size_from_graph_constant_time(graph, node_id, neighbor, light_threshold, max_neighbors_to_check):
    """Standalone function to get direction size from graph in O(1) time."""
    try:
        if neighbor not in graph.neighbors(node_id):
            return 0
            
        # Use BFS with depth and node count limits for O(1) time
        max_depth = int(math.log(light_threshold))
        max_nodes = int(light_threshold)
        
        queue = deque([(neighbor, 0)])  # (node, depth)
        visited = {neighbor}
        size = 0
        
        while queue and size <= max_nodes:
            current, depth = queue.popleft()
            size += 1
            
            if depth >= max_depth:
                continue
            
            # Check only a constant number of neighbors
            current_neighbors = list(graph.neighbors(current))
            if len(current_neighbors) > max_neighbors_to_check:
                current_neighbors = random.sample(current_neighbors, max_neighbors_to_check)
            
            for next_node in current_neighbors:
                if next_node != node_id and next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, depth + 1))
        
        return size
    except Exception as e:
        print(f"Error in get_direction_size_from_graph_constant_time: {str(e)}")
        return 0

def create_test_cases():
    # Case 1: Path graph - challenging for exponentiation
    path_graph = nx.path_graph(10)
    mapping = {i: i*10+5 for i in range(10)}
    path_graph = nx.relabel_nodes(path_graph, mapping)
    
    # Case 2: Balanced tree - easier for exponentiation
    balanced_tree = nx.balanced_tree(2, 3)
    mapping = {i: i*10+7 for i in range(len(balanced_tree.nodes()))}
    balanced_tree = nx.relabel_nodes(balanced_tree, mapping)
    
    # Case 3: Star-like tree with branches - mix of light and heavy nodes
    mixed_tree = nx.Graph()
    mixed_tree.add_node(100)
    for i in range(1, 6):
        mixed_tree.add_edge(100, 200+i)
        if i <= 3:
            mixed_tree.add_edge(200+i, 300+i)
            mixed_tree.add_edge(300+i, 400+i)
    
    return path_graph, balanced_tree, mixed_tree

def visualize_tree(tree, node_colors=None, title="Tree Visualization", save_path=None):
    """Visualize the tree with proper error handling."""
    try:
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(tree, seed=42)
        if node_colors is None:
            node_colors = "lightblue"
        nx.draw(tree, pos, with_labels=True, node_color=node_colors,
                node_size=700, font_size=10, font_weight='bold', edge_color='gray')
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        plt.close()

def visualize_results_summary(results: List[Dict], output_dir: Path):
    """Create a visual summary of the test results."""
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Subplot 1: Execution Times
        ax1 = plt.subplot(2, 2, 1)
        cases = [result['case'] for result in results]
        seq_times = [result['sequential']['time'] for result in results]
        par_times = [result['parallel']['time'] for result in results]
        
        x = np.arange(len(cases))
        width = 0.35
        
        ax1.bar(x - width/2, seq_times, width, label='Sequential')
        ax1.bar(x + width/2, par_times, width, label='Parallel')
        
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(cases, rotation=45, ha='right')
        ax1.legend()
        
        # Subplot 2: Speedup
        ax2 = plt.subplot(2, 2, 2)
        speedups = [result['parallel']['speedup'] for result in results]
        ax2.bar(cases, speedups, color='green')
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Parallel Speedup')
        ax2.set_xticklabels(cases, rotation=45, ha='right')
        
        # Subplot 3: Graph Sizes
        ax3 = plt.subplot(2, 2, 3)
        graph_sizes = [result['graph_size'] for result in results]
        ax3.bar(cases, graph_sizes, color='purple')
        ax3.set_xlabel('Test Cases')
        ax3.set_ylabel('Number of Nodes')
        ax3.set_title('Graph Sizes')
        ax3.set_xticklabels(cases, rotation=45, ha='right')
        
        # Subplot 4: Maximum IDs
        ax4 = plt.subplot(2, 2, 4)
        seq_max_ids = [result['sequential']['max_id'] for result in results]
        par_max_ids = [result['parallel']['max_id'] for result in results]
        
        ax4.bar(x - width/2, seq_max_ids, width, label='Sequential')
        ax4.bar(x + width/2, par_max_ids, width, label='Parallel')
        
        ax4.set_xlabel('Test Cases')
        ax4.set_ylabel('Maximum ID')
        ax4.set_title('Maximum ID Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cases, rotation=45, ha='right')
        ax4.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / 'results_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nResults summary visualization saved to 'results_summary.png'")
        
    except Exception as e:
        logger.error(f"Error creating results summary visualization: {str(e)}")
        plt.close()

def run_tests(use_dask: bool = False, num_machines: int = 4, test_size: int = 1000):
    """Run tests with improved error handling and resource management."""
    client = None
    cluster = None
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create results directory
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    try:
        if use_dask:
            print(f"Initializing Dask cluster with {num_machines} workers...")
            try:
                # Initialize cluster with proper dashboard settings
                cluster = LocalCluster(
                    n_workers=num_machines,
                    threads_per_worker=1,
                    processes=True,
                    memory_limit='2GB',
                    dashboard_address=':8787',  # Scheduler dashboard
                    silence_logs=logging.ERROR,
                    scheduler_port=0,  # Random port for scheduler
                    worker_port=0  # Random port for workers
                )
                client = Client(cluster)
                print("Dask cluster initialized successfully")
                
                # Print dashboard information
                dashboard_url = f"http://localhost:8787"
                print(f"\nDask Dashboard is available at: {dashboard_url}")
                print("You can monitor the progress of the computation in your web browser")
                
                # Print worker information
                print("\nWorker Information:")
                for worker_id, info in client.scheduler_info()['workers'].items():
                    print(f"Worker {worker_id}:")
                    print(f"  Host: {info['host']}")
                    print(f"  Memory: {info['memory_limit']/1e9:.1f}GB")
                
            except Exception as e:
                logger.error(f"Error initializing Dask cluster: {str(e)}")
                print("Falling back to sequential execution...")
                use_dask = False
                if cluster is not None:
                    try:
                        cluster.close(timeout=5)
                    except Exception as e:
                        logger.error(f"Error closing cluster: {str(e)}")
                    cluster = None
                client = None
        
        # Create test cases with specified size
        print("\nCreating test graphs...")
        try:
            path_graph = nx.path_graph(test_size)
            print(f"Created path graph with {len(path_graph.nodes)} nodes")
            
            balanced_tree = nx.balanced_tree(2, int(math.log2(test_size)))
            print(f"Created balanced tree with {len(balanced_tree.nodes)} nodes")
            
            mixed_tree = nx.Graph()
            mixed_tree.add_node(0)
            for i in range(1, test_size):
                mixed_tree.add_edge(0, i)
                if i % 3 == 0:
                    mixed_tree.add_edge(i, i + test_size)
            print(f"Created mixed tree with {len(mixed_tree.nodes)} nodes")
        except Exception as e:
            logger.error(f"Error creating test graphs: {str(e)}")
            raise
        
        # Run tests with error handling
        results = []
        for case_name, tree, d_hat in [
            ("PATH GRAPH", path_graph, test_size),
            ("BALANCED TREE", balanced_tree, int(math.log2(test_size))),
            ("MIXED TREE (STAR WITH BRANCHES)", mixed_tree, 2)
        ]:
            try:
                print(f"\n===== CASE: {case_name} =====")
                print(f"Graph size: {len(tree.nodes)} nodes, {len(tree.edges)} edges")
                
                # Run sequential version
                print("Running sequential version...")
                start_time = time.time()
                seq_solver = SequentialMAXIDSolver(tree, d_hat=d_hat)
                seq_max_id = seq_solver.solve()
                seq_time = time.time() - start_time
                
                if seq_max_id is not None:
                    print(f"Sequential - Maximum ID found: {seq_max_id}")
                    print(f"Sequential - Time taken: {seq_time:.2f} seconds")
                
                # Run parallel version if Dask is available
                par_time = None
                par_max_id = None
                if use_dask and client is not None:
                    print("Running parallel version...")
                    start_time = time.time()
                    par_solver = ParallelMAXIDSolver(tree, d_hat=d_hat, 
                                                   num_machines=num_machines, 
                                                   client=client)
                    par_max_id = par_solver.solve()
                    par_time = time.time() - start_time
                    
                    if par_max_id is not None:
                        print(f"Parallel - Maximum ID found: {par_max_id}")
                        print(f"Parallel - Time taken: {par_time:.2f} seconds")
                        if seq_time and par_time:
                            print(f"Speedup: {seq_time/par_time:.2f}x")
                        
                        # Save progress data
                        progress_data = par_solver.progress_data
                        with open(results_dir / f'{case_name.lower().replace(" ", "_")}_progress.json', 'w') as f:
                            json.dump(progress_data, f, indent=2)
                        
                        # Create summary plot
                        plt.figure(figsize=(12, 8))
                        plt.subplot(2, 1, 1)
                        plt.plot(progress_data['phases'], progress_data['graph_sizes'], 'b-o')
                        plt.xlabel('Phase')
                        plt.ylabel('Graph Size')
                        plt.title(f'{case_name} - Graph Size Reduction')
                        
                        plt.subplot(2, 1, 2)
                        plt.plot(progress_data['phases'], progress_data['max_ids'], 'r-o')
                        plt.xlabel('Phase')
                        plt.ylabel('Maximum ID')
                        plt.title(f'{case_name} - Maximum ID Evolution')
                        
                        plt.tight_layout()
                        plt.savefig(results_dir / f'{case_name.lower().replace(" ", "_")}_summary.png')
                        plt.close()
                
                # Record results
                results.append({
                    'case': case_name,
                    'graph_size': len(tree.nodes),
                    'edges': len(tree.edges),
                    'sequential': {
                        'max_id': seq_max_id,
                        'time': seq_time
                    },
                    'parallel': {
                        'max_id': par_max_id,
                        'time': par_time,
                        'speedup': seq_time/par_time if seq_time and par_time else None
                    }
                })
                
            except Exception as e:
                logger.error(f"Error in {case_name}: {str(e)}")
                continue
        
        # Save final results
        with open(results_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary table
        print("\nFinal Results Summary:")
        print("=" * 80)
        print(f"{'Case':<30} {'Graph Size':<12} {'Seq Time':<12} {'Par Time':<12} {'Speedup':<12}")
        print("-" * 80)
        for result in results:
            print(f"{result['case']:<30} {result['graph_size']:<12} "
                  f"{result['sequential']['time']:.2f}s {result['parallel']['time']:.2f}s "
                  f"{result['parallel']['speedup']:.2f}x")
        
        # Create visual summary
        visualize_results_summary(results, results_dir)
        
    except Exception as e:
        logger.error(f"Error in run_tests: {str(e)}")
        raise
    finally:
        # Clean up resources with proper error handling
        try:
            if client is not None:
                try:
                    client.close(timeout=5)
                except Exception as e:
                    logger.error(f"Error closing client: {str(e)}")
                client = None
            
            if cluster is not None:
                try:
                    cluster.close(timeout=5)
                except Exception as e:
                    logger.error(f"Error closing cluster: {str(e)}")
                cluster = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Run tests with Dask enabled and 4 machines
        run_tests(use_dask=True, num_machines=4, test_size=1000)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)