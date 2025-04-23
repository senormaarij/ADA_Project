import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Set this before importing pyplot
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import math
from collections import deque
import dask
from dask.distributed import Client, LocalCluster
import numpy as np
import logging
import pickle
import cloudpickle
import concurrent.futures
import multiprocessing
from typing import Dict, Set, List, Optional, Tuple, Any
import random
import time
import os
import sys
from pathlib import Path
import threading
from queue import Queue
import signal
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.exit_handler = GracefulExit()
        self._initialize_nodes()

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
        """Execute the CompressLightSubTrees procedure."""
        iterations = int(math.log2(self.d_hat)) + 2
        max_iterations = 5  # Limit iterations to prevent infinite loops
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        for i in range(min(iterations, max_iterations)):
            if time.time() - start_time > timeout:
                print(f"Compression timeout after {i} iterations")
                break
            
            print(f"Compression iteration {i+1}/{min(iterations, max_iterations)}")
            compression_map = {}
            active_nodes = [v for v, node in self.nodes.items() if node['state'] == "active"]
            
            if not active_nodes:
                print("No active nodes left, stopping compression")
                break
            
            print(f"Processing {len(active_nodes)} active nodes...")
            
            # Process nodes sequentially
            for node_id in active_nodes:
                if self.exit_handler.should_exit():
                    return self._create_compressed_graph(compression_map)
                
                result = self._process_node_compression(node_id)
                v, state, target = result
                if state == "happy" and target is not None:
                    compression_map[v] = target
                    self.nodes[target]['compressed_nodes'].add(v)
                    self.nodes[v]['compressed_into'] = target
                    self.nodes[v]['state'] = "happy"
            
            self.compression_maps.append(compression_map)
            self._propagate_max_ids()

            if all(node['state'] != "active" for node in self.nodes.values()):
                print("All nodes processed, stopping compression")
                break

        return self._create_compressed_graph(compression_map)

    def compress_paths(self, G: nx.Graph) -> nx.Graph:
        """Execute the CompressPaths procedure."""
        G_prime = G.copy()
        path_nodes = [node for node, degree in G.degree() if degree == 2]
        compression_map = {}
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        print(f"Starting path compression with {len(path_nodes)} nodes to process")
        
        while path_nodes and time.time() - start_time <= timeout:
            if self.exit_handler.should_exit():
                break
            
            v = path_nodes[0]
            path_nodes.pop(0)
            
            # Check if node exists and has degree 2
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
                
                # Check if new degree-2 nodes were created
                if u in G_prime.nodes() and G_prime.degree(u) == 2 and u not in path_nodes:
                    path_nodes.append(u)
                if w in G_prime.nodes() and G_prime.degree(w) == 2 and w not in path_nodes:
                    path_nodes.append(w)
            
            if len(path_nodes) % 100 == 0:
                print(f"Processed {len(path_nodes)} path nodes...")

        if time.time() - start_time > timeout:
            print("Path compression timeout reached")
        
        # After compression, synchronize self.nodes with the graph
        removed_nodes = set(G.nodes()) - set(G_prime.nodes())
        for node in removed_nodes:
            if node in self.nodes:
                self.nodes[node]['state'] = "compressed"
        
        self.compression_maps.append(compression_map)
        return G_prime

    def decompress(self):
        """Reverse the compression process to assign max IDs to all nodes."""
        # Process compression maps in reverse order
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

    def _process_node_compression(self, node_id: int) -> Tuple[int, str, Optional[int]]:
        """Process a single node for compression."""
        try:
            # Check if node exists in the current graph
            current_graph = self.tree.to_nx()
            if node_id not in current_graph.nodes():
                logger.warning(f"Node {node_id} not in current graph, skipping")
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

            # Process the node with timeout
            start_time = time.time()
            timeout = 5  # 5 second timeout per node
            
            # Get full directions and largest direction
            fullDirs, largestDir = self._probe_directions_local(local_node)
            if time.time() - start_time > timeout:
                print(f"Timeout processing node {node_id}")
                return node_id, "active", None
            
            # Handle compression based on full directions
            if isinstance(fullDirs, set) and len(fullDirs) >= 2:
                return node_id, "sad", None
            elif isinstance(fullDirs, set) and len(fullDirs) == 1:
                target = next(iter(fullDirs))  # Get the single full direction
                self._exponentiate_local(local_node, exclude_dirs={target})
            elif isinstance(largestDir, int):
                self._exponentiate_local(local_node, exclude_dirs={largestDir})

            # Check for light subtrees
            for u in local_node['neighbors']:
                if time.time() - start_time > timeout:
                    print(f"Timeout processing node {node_id}")
                    return node_id, "active", None
                
                # Check if neighbor exists in current graph
                if u not in current_graph.nodes():
                    continue
                    
                is_light, target = self._is_light_local(local_node, u)
                if is_light and isinstance(target, int):
                    opposite = self._get_opposite_direction_local(local_node, u)
                    if isinstance(opposite, set) and opposite.issubset(local_node['sv']):
                        return node_id, "happy", u
            
            if len(local_node['sv']) >= 2 * self.n ** (self.delta/4):
                return node_id, "full", None

            return node_id, "active", None
        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            return node_id, "active", None

    def _probe_directions_local(self, node: Dict[str, Any]) -> Tuple[Set[int], Optional[int]]:
        """Local version of probe_directions for serializable processing."""
        try:
            B_values = {}
            for u in node['neighbors']:
                B_values[u] = 0
                for w in node['sv']:
                    if w == node['id']:
                        continue
                    rv_w = self._get_rv_mapping_local(node, w)
                    if rv_w == u:
                        # Check if w exists in self.nodes
                        if w not in self.nodes:
                            continue
                            
                        rw_v = self._get_rv_mapping_local(self.nodes[w], node['id'])
                        if rw_v is not None:
                            away_size = len(self._get_opposite_direction_local(self.nodes[w], rw_v))
                            B_values[u] += away_size

            fullDirs = {u for u, b in B_values.items() if b >= self.light_threshold * 8}
            largestDir = None
            if not fullDirs and B_values:
                largestDir = max(B_values.items(), key=lambda x: x[1])[0]
            return fullDirs, largestDir
        except Exception as e:
            logger.error(f"Error in probe_directions_local: {str(e)}")
            return set(), None

    def _get_rv_mapping_local(self, node: Dict[str, Any], target: int) -> Optional[int]:
        """Local version of get_rv_mapping for serializable processing."""
        try:
            if node['id'] == target:
                return None
            queue = deque([(node['id'], None)])
            visited = {node['id']}
            parent = {}
            while queue:
                current, prev = queue.popleft()
                if current == target:
                    while current in parent and parent[current] != node['id']:
                        current = parent[current]
                    return current
                    
                # Check if current node exists in self.nodes
                if current not in self.nodes:
                    continue
                    
                for neighbor in self.nodes[current]['neighbors']:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        queue.append((neighbor, current))
            return None
        except Exception as e:
            logger.error(f"Error in get_rv_mapping_local: {str(e)}")
            return None

    def _get_opposite_direction_local(self, node: Dict[str, Any], neighbor: int) -> Set[int]:
        """Local version of get_opposite_direction for serializable processing."""
        try:
            all_nodes = set(self.tree.to_nx().nodes())
            direction_nodes = self._get_direction_local(node, neighbor)
            return all_nodes - direction_nodes
        except Exception as e:
            logger.error(f"Error in get_opposite_direction_local: {str(e)}")
            return set()

    def _get_direction_local(self, node: Dict[str, Any], neighbor: int) -> Set[int]:
        """Local version of get_direction for serializable processing."""
        try:
            if neighbor not in node['neighbors']:
                return set()
                
            # Get a fresh copy of the graph
            subgraph = self.tree.to_nx().copy()
            
            # Check if both nodes exist in the graph
            if node['id'] not in subgraph.nodes() or neighbor not in subgraph.nodes():
                return set()
                
            # Check if the edge exists
            if not subgraph.has_edge(node['id'], neighbor):
                return set()
                
            subgraph.remove_edge(node['id'], neighbor)
            for component in nx.connected_components(subgraph):
                if neighbor in component:
                    return component
            return set()
        except Exception as e:
            logger.error(f"Error in get_direction_local: {str(e)}")
            return set()

    def _is_light_local(self, node: Dict[str, Any], neighbor: int) -> Tuple[bool, Optional[int]]:
        """Local version of is_light for serializable processing."""
        try:
            away_nodes = self._get_opposite_direction_local(node, neighbor)
            if len(away_nodes) <= self.light_threshold:
                return True, neighbor
            return False, None
        except Exception as e:
            logger.error(f"Error in is_light_local: {str(e)}")
            return False, None

    def _exponentiate_local(self, node: Dict[str, Any], exclude_dirs: Optional[Set[int]] = None) -> None:
        """Local version of exponentiate for serializable processing."""
        try:
            exclude_dirs = exclude_dirs or set()
            for u in node['neighbors'] - exclude_dirs:
                for w in list(node['sv']):
                    if w == node['id']:
                        continue
                        
                    # Check if w exists in self.nodes
                    if w not in self.nodes:
                        continue
                        
                    rv_w = self._get_rv_mapping_local(node, w)
                    if rv_w == u:
                        rw_v = self._get_rv_mapping_local(self.nodes[w], node['id'])
                        if rw_v is not None:
                            w_away = self._get_opposite_direction_local(self.nodes[w], rw_v) - {w}
                            node['sv'].update(w_away)
        except Exception as e:
            logger.error(f"Error in exponentiate_local: {str(e)}")

    def _create_compressed_graph(self, compression_map: Dict[int, Any]) -> nx.Graph:
        """Create a compressed graph from a compression map."""
        G_prime = self.tree.to_nx()
        for v, target in compression_map.items():
            if v not in G_prime.nodes():
                continue  # Skip if node already removed
                
            if isinstance(target, tuple):  # Path compression
                u, w = target
                if v in G_prime.nodes() and u in G_prime.nodes() and w in G_prime.nodes():
                    G_prime.add_edge(u, w)
                    G_prime.remove_node(v)
            else:  # Subtree compression
                if v in G_prime.nodes():
                    G_prime.remove_node(v)
        return G_prime

    def _propagate_max_ids(self):
        """Propagate maximum IDs across the tree."""
        changed = True
        while changed:
            changed = False
            for v, node in self.nodes.items():
                old_max = node['max_id']
                for w in node['sv']:
                    if w in self.nodes and self.nodes[w]['max_id'] > node['max_id']:
                        node['max_id'] = self.nodes[w]['max_id']
                        changed = True
                for w in node['compressed_nodes']:
                    if w in self.nodes and self.nodes[w]['max_id'] > node['max_id']:
                        node['max_id'] = self.nodes[w]['max_id']
                        changed = True

class SequentialMAXIDSolver(MAXIDSolver):
    """Sequential implementation of MAX-ID algorithm."""
    def solve(self):
        """Solve the MAX-ID problem sequentially."""
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
                    G.append(G_next)
                    phases += 1
                    
                    print(f"Phase {phases}: Graph size reduced to {len(G_next.nodes)} nodes")
                    
                    # Visualize progress safely
                    self._visualize_progress(phases, G_next)
                    
                    if len(G_next.nodes) == 1:
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
    """Parallel implementation of MAX-ID algorithm using Dask."""
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
            'max_ids': []
        }

    def _process_machine_compression(self, machine_id: int, partition: Set[int], 
                                   boundary_nodes: Set[int], graph: nx.Graph) -> Dict[int, int]:
        """Process compression for a specific machine's partition."""
        try:
            # Create a local copy of nodes to avoid serializing the entire self object
            local_nodes = {}
            for node_id in partition:
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
            active_nodes = [v for v in partition if v in local_nodes and local_nodes[v]['state'] == "active"]
            
            for node_id in active_nodes:
                # Check if node exists in the graph
                if node_id not in graph.nodes():
                    continue
                    
                # Simplified processing for distributed execution
                # Check for light subtrees directly
                for u in local_nodes[node_id]['neighbors']:
                    # Check if neighbor exists in the graph
                    if u not in graph.nodes():
                        continue
                        
                    # Calculate direction size directly from the graph
                    direction_size = self._get_direction_size_from_graph(graph, node_id, u)
                    if direction_size <= self.light_threshold:
                        compression_map[node_id] = u
                        break
            
            return compression_map
        except Exception as e:
            logger.error(f"Error in machine {machine_id} compression: {str(e)}")
            return {}
            
    def _get_direction_size_from_graph(self, graph: nx.Graph, node_id: int, neighbor: int) -> int:
        """Get the size of a direction using only the provided graph."""
        try:
            if neighbor not in graph.neighbors(node_id):
                return 0
            subgraph = graph.copy()
            subgraph.remove_edge(node_id, neighbor)
            for component in nx.connected_components(subgraph):
                if neighbor in component:
                    return len(component)
            return 0
        except Exception as e:
            logger.error(f"Error in get_direction_size_from_graph: {str(e)}")
            return 0

    def solve(self):
        """Solve the MAX-ID problem in parallel using Dask."""
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
                            for node_id in partition:
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
                                self.light_threshold
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
                    
                    # Create compressed graph
                    G_prime = self._create_compressed_graph(compression_map)
                    compression_time = time.time() - phase_start
                    
                    if self.exit_handler.should_exit():
                        break
                    
                    # Path compression
                    path_start = time.time()
                    G_next = self.compress_paths(G_prime)
                    path_time = time.time() - path_start
                    
                    G.append(G_next)
                    phases += 1
                    
                    # Record progress data
                    self.progress_data['phases'].append(phases)
                    self.progress_data['graph_sizes'].append(len(G_next.nodes))
                    self.progress_data['compression_times'].append(compression_time)
                    self.progress_data['path_compression_times'].append(path_time)
                    self.progress_data['max_ids'].append(max(node['max_id'] for node in self.nodes.values()))
                    
                    print(f"Phase {phases}: Graph size reduced to {len(G_next.nodes)} nodes")
                    print(f"Compression time: {compression_time:.2f}s, Path compression time: {path_time:.2f}s")
                    
                    # Visualize progress
                    self._visualize_progress(phases, G_next)
                    
                    if len(G_next.nodes) == 1:
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
def process_machine_compression_standalone(machine_id, partition, boundary_nodes, graph, local_nodes, light_threshold):
    """Standalone function for processing compression in a separate process."""
    try:
        compression_map = {}
        active_nodes = [v for v in partition if v in local_nodes and local_nodes[v]['state'] == "active"]
        
        for node_id in active_nodes:
            # Check if node exists in the graph
            if node_id not in graph.nodes():
                continue
                
            # Simplified processing for distributed execution
            # Check for light subtrees directly
            for u in local_nodes[node_id]['neighbors']:
                # Check if neighbor exists in the graph
                if u not in graph.nodes():
                    continue
                    
                # Calculate direction size directly from the graph
                direction_size = get_direction_size_from_graph_standalone(graph, node_id, u)
                if direction_size <= light_threshold:
                    compression_map[node_id] = u
                    break
        
        return compression_map
    except Exception as e:
        print(f"Error in machine {machine_id} compression: {str(e)}")
        return {}

def get_direction_size_from_graph_standalone(graph, node_id, neighbor):
    """Standalone function to get direction size from graph."""
    try:
        if neighbor not in graph.neighbors(node_id):
            return 0
        subgraph = graph.copy()
        subgraph.remove_edge(node_id, neighbor)
        for component in nx.connected_components(subgraph):
            if neighbor in component:
                return len(component)
        return 0
    except Exception as e:
        print(f"Error in get_direction_size_from_graph_standalone: {str(e)}")
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
