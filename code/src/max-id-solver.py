"""
Implementation of the MAX-ID algorithm from "Optimal Deterministic Massively Parallel Connectivity on Forests"
by Balliu et al.

This implementation focuses on the MAX-ID problem for a single tree, following the paper's pseudocode
and includes all necessary helper functions and subroutines.
"""

import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import defaultdict, deque
import time
import random


class MPCNode:
    """
    Represents a node in the graph with its local memory and state.
    Each node maintains its view (knowledge) of the graph.
    """
    def __init__(self, id):
        self.id = id                  # Node identifier
        self.neighbors = set()        # Direct neighbors
        self.sv = set()               # Current view/knowledge set
        self.state = "active"         # One of: active, happy, full, sad
        self.max_id = id              # Current known maximum ID
    
    def add_neighbor(self, neighbor_id):
        """Add a neighbor to this node"""
        self.neighbors.add(neighbor_id)
    
    def initialize_sv(self):
        """Initialize knowledge set with direct neighbors and self"""
        self.sv = self.neighbors.copy()
        self.sv.add(self.id)


class MaxIDSolver:
    """
    Implementation of the MAX-ID algorithm from the paper.
    Solves the problem of finding the maximum ID in a tree.
    """
    def __init__(self, tree, delta=0.25):
        """
        Initialize the MAX-ID solver with a tree.
        
        Args:
            tree: A networkx Graph representing a tree
            delta: Parameter for defining light/heavy nodes (nδ/8)
        """
        self.tree = tree
        self.delta = delta
        self.nodes = {}
        self.light_threshold = math.pow(len(tree.nodes), delta/8)
        
        # Convert networkx graph to our node representation
        for node_id in tree.nodes():
            self.nodes[node_id] = MPCNode(node_id)
        
        for u, v in tree.edges():
            self.nodes[u].add_neighbor(v)
            self.nodes[v].add_neighbor(u)
        
        # Initialize node views
        for node in self.nodes.values():
            node.initialize_sv()
    
    def is_light(self, node_id, avoid_neighbor=None):
        """
        Check if node is light against some neighbor.
        A node is light if there exists a neighbor such that 
        the subtree away from that neighbor has size ≤ threshold.
        
        Args:
            node_id: ID of the node to check
            avoid_neighbor: If specified, check only against this neighbor
            
        Returns:
            True if the node is light, False otherwise
        """
        node = self.nodes[node_id]
        neighbors_to_check = [avoid_neighbor] if avoid_neighbor else node.neighbors
        
        for neighbor in neighbors_to_check:
            if neighbor not in node.neighbors:
                continue
                
            # Compute size of subtree away from this neighbor
            away_size = self._estimate_subtree_size(node_id, neighbor)
            if away_size <= self.light_threshold:
                return True
        
        return False
    
    def _estimate_subtree_size(self, node_id, avoid_neighbor):
        """
        Estimate the size of the subtree reachable from node_id
        without going through avoid_neighbor.
        
        This simulates computing G_v→u in the paper.
        
        Args:
            node_id: Starting node
            avoid_neighbor: Neighbor to avoid
            
        Returns:
            Size of the subtree
        """
        visited = {node_id, avoid_neighbor}
        queue = deque()
        
        # Add all neighbors except the one to avoid
        for neighbor in self.nodes[node_id].neighbors:
            if neighbor != avoid_neighbor:
                queue.append(neighbor)
        
        size = 1  # Count starting node
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            size += 1
            
            for neighbor in self.nodes[current].neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return size
    
    def get_paths_to(self, node_id, target_ids):
        """
        Get the next hop on the path from node_id to each target_id.
        Used to determine in which direction nodes should be found.
        
        Args:
            node_id: Starting node ID
            target_ids: Set of target node IDs
            
        Returns:
            Dictionary mapping target_id to next_hop
        """
        result = {}
        visited = {node_id}
        queue = deque([(node_id, None)])
        
        while queue and len(result) < len(target_ids):
            current, prev = queue.popleft()
            
            if current in target_ids:
                # Reconstruct path from node_id to current
                result[current] = prev
            
            for neighbor in self.nodes[current].neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current))
        
        return result
    
    def probe_directions(self, node_id):
        """
        Implementation of the ProbeDirections subroutine from the paper.
        Estimates how many nodes would be learned by exponentiating in each direction.
        
        Args:
            node_id: ID of the node performing the probe
            
        Returns:
            fullDirs: Set of directions that would exceed the threshold
            largestDir: Direction with the most nodes (if fullDirs is empty)
        """
        node = self.nodes[node_id]
        
        # Compute Bv→u for each neighbor u
        B_values = {}
        for neighbor in node.neighbors:
            # Sum of |Sw¬→rw(v)| for all w in Sv→u
            B_values[neighbor] = 0
            
            for w_id in node.sv:
                w = self.nodes[w_id]
                if w_id == node_id:
                    continue
                    
                # Check if w is in the direction of neighbor
                paths = self.get_paths_to(node_id, {w_id})
                if w_id in paths and paths[w_id] == neighbor:
                    # Estimate |Sw¬→rw(v)|
                    away_size = self._estimate_subtree_size(w_id, node_id)
                    B_values[neighbor] += away_size
        
        # Identify fullDirs (directions that would exceed threshold if exponentiated)
        fullDirs = {u for u, b in B_values.items() if b >= self.light_threshold * 8}
        
        # If fullDirs is empty, identify largestDir
        largestDir = None
        if not fullDirs:
            largestDir = max(B_values.items(), key=lambda x: x[1])[0] if B_values else None
        
        return fullDirs, largestDir
    
    def exponentiate(self, node_id, exclude_dirs=None):
        """
        Perform the Exp operation as defined in the paper.
        Node learns information from its neighbors in specified directions.
        
        Args:
            node_id: ID of the node performing exponentiation
            exclude_dirs: Set of directions to exclude
            
        Returns:
            Set of newly learned nodes
        """
        node = self.nodes[node_id]
        exclude_dirs = exclude_dirs or set()
        new_nodes = set()
        
        # For each neighbor not in exclude_dirs
        for neighbor in node.neighbors - exclude_dirs:
            # For each node w in Sv→u
            # In this simulation, we determine Sv→u by checking if w is reachable via neighbor
            potential_nodes = set()
            for w_id in node.sv:
                if w_id == node_id:
                    continue
                
                paths = self.get_paths_to(node_id, {w_id})
                if w_id in paths and paths[w_id] == neighbor:
                    potential_nodes.add(w_id)
            
            # For each node w in potential_nodes, learn Sw¬→rw(v)
            for w_id in potential_nodes:
                w = self.nodes[w_id]
                # Determine rw(v) - the neighbor of w on the path to v
                paths = self.get_paths_to(w_id, {node_id})
                if node_id in paths:
                    rw_v = paths[node_id]
                    
                    # Add all nodes in w's view except those in the direction of v
                    for x_id in w.sv:
                        x = self.nodes[x_id]
                        x_paths = self.get_paths_to(w_id, {x_id})
                        if x_id in x_paths and x_paths[x_id] != rw_v:
                            new_nodes.add(x_id)
        
        # Update node's view
        node.sv.update(new_nodes)
        return new_nodes
    
    def compress_light_subtrees(self, iterations=None):
        """
        Implementation of the CompressLightSubTrees subroutine from the paper.
        Compresses light subtrees into their closest heavy node.
        
        Args:
            iterations: Number of iterations to run, defaults to log(diameter)
            
        Returns:
            Dictionary mapping node IDs to states
        """
        # If iterations not specified, estimate based on tree diameter
        if iterations is None:
            try:
                diameter = nx.diameter(self.tree)
                iterations = int(math.log2(diameter)) + 2
            except:
                # Fallback if diameter can't be computed
                iterations = int(math.log2(len(self.nodes))) + 2
        
        print(f"Running CompressLightSubTrees for {iterations} iterations...")
        
        # Main iterative algorithm
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Process nodes in parallel (simulated)
            updates = []
            
            for node_id, node in self.nodes.items():
                if node.state != "active":
                    continue
                
                # Probe directions
                fullDirs, largestDir = self.probe_directions(node_id)
                
                # Determine which directions to exponentiate
                if len(fullDirs) >= 2:
                    # If multiple directions are too large, node is heavy
                    updates.append((node_id, {"state": "sad"}))
                    continue
                    
                elif len(fullDirs) == 1:
                    # Exponentiate in all directions except fullDirs
                    new_nodes = self.exponentiate(node_id, exclude_dirs=fullDirs)
                    updates.append((node_id, {"sv_update": new_nodes}))
                    
                elif largestDir is not None:
                    # Exponentiate in all directions except largestDir
                    new_nodes = self.exponentiate(node_id, exclude_dirs={largestDir})
                    updates.append((node_id, {"sv_update": new_nodes}))
                
                # Check if node becomes happy
                for neighbor in node.neighbors:
                    if self.is_light(node_id, neighbor):
                        subtree_size = self._estimate_subtree_size(node_id, neighbor)
                        # Check if the subtree is fully contained in the node's view
                        if len(node.sv) >= subtree_size:
                            updates.append((node_id, {"state": "happy"}))
                            break
                
                # Check if node becomes full
                if len(node.sv) >= 2 * len(self.nodes) ** (self.delta/4):
                    updates.append((node_id, {"state": "full"}))
            
            # Apply updates (simulating parallel execution)
            for node_id, update in updates:
                if "state" in update:
                    self.nodes[node_id].state = update["state"]
                if "sv_update" in update:
                    self.nodes[node_id].sv.update(update["sv_update"])
            
            # Update max_id values
            self._propagate_max_ids()
            
            # Print statistics
            states = {"active": 0, "happy": 0, "sad": 0, "full": 0}
            for node in self.nodes.values():
                states[node.state] += 1
            print(f"  States: {states}")
            
            # Check termination condition
            if states["active"] == 0:
                print("All nodes have decided their state, terminating early.")
                break
        
        # Post-process to determine the maximum ID
        self._propagate_max_ids()
        
        # Final output - map each node to its state
        return {node_id: node.state for node_id, node in self.nodes.items()}
    
    def _propagate_max_ids(self):
        """
        Propagate maximum IDs across the tree.
        Each node updates its max_id based on its neighbors.
        """
        changed = True
        while changed:
            changed = False
            for node_id, node in self.nodes.items():
                old_max = node.max_id
                
                # Update max_id based on all nodes in view
                for other_id in node.sv:
                    other = self.nodes[other_id]
                    if other.max_id > node.max_id:
                        node.max_id = other.max_id
                        changed = True
    
    def solve(self):
        """
        Solve the MAX-ID problem for the tree.
        
        Returns:
            Dictionary mapping node ID to max ID
        """
        # Classify nodes as light/heavy
        light_nodes = set()
        heavy_nodes = set()
        
        for node_id in self.nodes:
            if self.is_light(node_id):
                light_nodes.add(node_id)
            else:
                heavy_nodes.add(node_id)
        
        print(f"Initial classification: {len(light_nodes)} light nodes, {len(heavy_nodes)} heavy nodes")
        
        # Run the algorithm
        self.compress_light_subtrees()
        
        # Return the result
        return {node_id: node.max_id for node_id, node in self.nodes.items()}


def visualize_tree(tree, node_colors=None, node_labels=None, title="Tree Visualization"):
    """
    Visualize a tree with optional node colors and labels.
    
    Args:
        tree: A networkx Graph
        node_colors: Dictionary mapping node ID to color
        node_labels: Dictionary mapping node ID to label
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(tree, seed=42)
    
    # Default node color if not provided
    if node_colors is None:
        node_colors = {node: 'skyblue' for node in tree.nodes()}
    
    # Extract colors in the order of nodes
    colors = [node_colors[node] for node in tree.nodes()]
    
    # Draw the tree
    nx.draw_networkx_edges(tree, pos)
    nx.draw_networkx_nodes(tree, pos, node_color=colors, node_size=500, alpha=0.8)
    
    # If labels are provided, draw them
    if node_labels:
        nx.draw_networkx_labels(tree, pos, labels=node_labels)
    else:
        nx.draw_networkx_labels(tree, pos)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_path_graph(n):
    """Create a path graph with n nodes"""
    G = nx.path_graph(n)
    # Relabel nodes with random IDs
    mapping = {i: random.randint(100, 999) for i in range(n)}
    return nx.relabel_nodes(G, mapping)


def create_balanced_tree(height, branching_factor):
    """Create a balanced tree with given height and branching factor"""
    G = nx.balanced_tree(branching_factor, height)
    # Relabel nodes with random IDs
    mapping = {i: random.randint(100, 999) for i in range(len(G.nodes()))}
    return nx.relabel_nodes(G, mapping)


def create_star_with_branches(center_branches, end_branches):
    """Create a star graph with additional branches at the ends"""
    G = nx.Graph()
    
    # Create star with center_branches
    node_id = 0
    center = random.randint(900, 999)  # High ID for center
    G.add_node(center)
    
    for i in range(center_branches):
        branch_start = random.randint(100, 899)
        G.add_edge(center, branch_start)
        
        # Add branches to some endpoints
        if i < end_branches:
            current = branch_start
            branch_length = random.randint(2, 4)
            
            for j in range(branch_length):
                next_node = random.randint(100, 899)
                G.add_edge(current, next_node)
                current = next_node
    
    return G


def test_max_id_solver():
    """Test the MAX-ID solver on three different tree topologies"""
    
    # Case 1: Path Graph (worst case for exponentiation)
    print("\n===== CASE 1: PATH GRAPH =====")
    path_graph = create_path_graph(15)
    max_id = max(path_graph.nodes())
    
    print(f"Path graph with {len(path_graph.nodes())} nodes, max ID: {max_id}")
    
    solver = MaxIDSolver(path_graph)
    result = solver.solve()
    
    # Verify result
    correct = all(node_max_id == max_id for node_max_id in result.values())
    print(f"All nodes found correct max ID: {correct}")
    
    # Visualize result
    node_colors = {}
    for node_id, node in solver.nodes.items():
        if node.state == "happy":
            node_colors[node_id] = "lightgreen"
        elif node.state == "sad":
            node_colors[node_id] = "salmon"
        elif node.state == "full":
            node_colors[node_id] = "purple"
        else:
            node_colors[node_id] = "lightblue"
    
    node_labels = {node_id: f"{node_id}\n(Max: {result[node_id]})" for node_id in path_graph.nodes()}
    visualize_tree(path_graph, node_colors, node_labels, "Path Graph MAX-ID Result")
    
    # Case 2: Balanced Tree (best case for exponentiation)
    print("\n===== CASE 2: BALANCED TREE =====")
    balanced_tree = create_balanced_tree(3, 2)  # Height 3, branching factor 2
    max_id = max(balanced_tree.nodes())
    
    print(f"Balanced tree with {len(balanced_tree.nodes())} nodes, max ID: {max_id}")
    
    solver = MaxIDSolver(balanced_tree)
    result = solver.solve()
    
    # Verify result
    correct = all(node_max_id == max_id for node_max_id in result.values())
    print(f"All nodes found correct max ID: {correct}")
    
    # Visualize result
    node_colors = {}
    for node_id, node in solver.nodes.items():
        if node.state == "happy":
            node_colors[node_id] = "lightgreen"
        elif node.state == "sad":
            node_colors[node_id] = "salmon"
        elif node.state == "full":
            node_colors[node_id] = "purple"
        else:
            node_colors[node_id] = "lightblue"
    
    node_labels = {node_id: f"{node_id}\n(Max: {result[node_id]})" for node_id in balanced_tree.nodes()}
    visualize_tree(balanced_tree, node_colors, node_labels, "Balanced Tree MAX-ID Result")
    
    # Case 3: Star with Branches (mix of light and heavy nodes)
    print("\n===== CASE 3: STAR WITH BRANCHES =====")
    star_tree = create_star_with_branches(6, 3)
    max_id = max(star_tree.nodes())
    
    print(f"Star with branches: {len(star_tree.nodes())} nodes, max ID: {max_id}")
    
    solver = MaxIDSolver(star_tree)
    result = solver.solve()
    
    # Verify result
    correct = all(node_max_id == max_id for node_max_id in result.values())
    print(f"All nodes found correct max ID: {correct}")
    
    # Visualize result
    node_colors = {}
    for node_id, node in solver.nodes.items():
        if node.state == "happy":
            node_colors[node_id] = "lightgreen"
        elif node.state == "sad":
            node_colors[node_id] = "salmon"
        elif node.state == "full":
            node_colors[node_id] = "purple"
        else:
            node_colors[node_id] = "lightblue"
    
    node_labels = {node_id: f"{node_id}\n(Max: {result[node_id]})" for node_id in star_tree.nodes()}
    visualize_tree(star_tree, node_colors, node_labels, "Star with Branches MAX-ID Result")


if __name__ == "__main__":
    test_max_id_solver()
