import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import defaultdict, deque

class MAXIDSolver:
    """
    Implementation of the MAX-ID algorithm from the paper.
    This solves the problem of finding the maximum ID in a tree.
    """
    def __init__(self, tree, d_hat, delta=0.25):
        """
        Initialize the MAX-ID solver with a tree.
        
        Args:
            tree: A networkx Graph representing a tree
            d_hat: An upper bound on the diameter of the tree (d_hat >= diam(G))
            delta: Parameter for defining light/heavy nodes (n^δ/8)
        """
        self.tree = tree
        self.d_hat = d_hat
        self.delta = delta
        self.n = len(tree.nodes)
        self.nodes = {}
        self.light_threshold = math.pow(self.n, delta/8)
        
        # Convert networkx graph to our node representation
        for node_id in tree.nodes():
            self.nodes[node_id] = {
                'id': node_id,
                'neighbors': set(tree.neighbors(node_id)),
                'sv': set(),  # Knowledge set
                'state': "active",  # One of: active, happy, full, sad
                'max_id': node_id  # Current known maximum ID
            }
            
        # Initialize node views (Sv)
        for node in self.nodes.values():
            node['sv'] = set(node['neighbors'])
            node['sv'].add(node['id'])

    def get_direction(self, v, u):
        """
        Get all nodes in the direction of u from v (Gv→u)
        
        Args:
            v: Source node ID
            u: Neighbor node ID
            
        Returns:
            Set of nodes in the direction of u from v
        """
        if u not in self.nodes[v]['neighbors']:
            return set()
            
        # Create a subgraph by removing the edge between v and u
        subgraph = self.tree.copy()
        subgraph.remove_edge(v, u)
        
        # Find the connected component containing u
        for component in nx.connected_components(subgraph):
            if u in component:
                return component
        return set()
    
    def get_opposite_direction(self, v, u):
        """
        Get all nodes in the direction away from u from v (Gv ̸→u)
        
        Args:
            v: Source node ID
            u: Neighbor node ID
            
        Returns:
            Set of nodes in direction away from u (including v)
        """
        all_nodes = set(self.tree.nodes())
        direction_nodes = self.get_direction(v, u)
        return all_nodes - direction_nodes
    
    def is_light(self, v, against_neighbor=None):
        """
        Check if node is light (has a subtree of size <= threshold)
        
        Args:
            v: Node ID to check
            against_neighbor: If specified, check only against this neighbor
            
        Returns:
            (is_light, neighbor) tuple where neighbor is the one v is light against
        """
        neighbors = [against_neighbor] if against_neighbor else self.nodes[v]['neighbors']
        
        for neighbor in neighbors:
            # Get nodes in direction away from neighbor (Gv ̸→neighbor)
            away_nodes = self.get_opposite_direction(v, neighbor)
            if len(away_nodes) <= self.light_threshold:
                return True, neighbor
        return False, None
    
    def get_rv_mapping(self, v, w):
        """
        For node v, find the neighbor that leads to node w (rv(w))
        
        Args:
            v: Source node
            w: Target node
            
        Returns:
            Neighbor of v on the path to w or None if v=w or not connected
        """
        if v == w:
            return None
        
        # Perform BFS to find the path from v to w
        queue = deque([(v, None)])
        visited = {v}
        parent = {}
        
        while queue:
            node, prev = queue.popleft()
            if node == w:
                # Reconstruct the first step from v
                current = node
                while current in parent and parent[current] != v:
                    current = parent[current]
                return current
            
            for neighbor in self.nodes[node]['neighbors']:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append((neighbor, node))
        
        return None  # No path found
    
    def probe_directions(self, v):
        """
        Probe directions to determine where to exponentiate
        
        Args:
            v: Node ID
            
        Returns:
            fullDirs: Set of directions that would exceed the threshold
            largestDir: Direction with the most nodes (if fullDirs is empty)
        """
        node = self.nodes[v]
        
        # For each neighbor u, compute Bv→u
        B_values = {}
        for u in node['neighbors']:
            B_values[u] = 0
            for w in node['sv']:
                if w == v:
                    continue
                
                # Check if w is in the direction of u
                rv_w = self.get_rv_mapping(v, w)
                if rv_w == u:  # If u is on the path from v to w
                    # Get Sw ̸→rw(v)
                    rw_v = self.get_rv_mapping(w, v)
                    if rw_v is not None:  # If there is a path from w to v
                        away_size = len(self.get_opposite_direction(w, rw_v))
                        B_values[u] += away_size
        
        # Identify fullDirs
        fullDirs = {u for u, b in B_values.items() if b >= self.light_threshold * 8}
        
        # If fullDirs is empty, identify largestDir
        largestDir = None
        if not fullDirs and B_values:
            largestDir = max(B_values.items(), key=lambda x: x[1])[0]
        
        return fullDirs, largestDir
    
    def exponentiate(self, v, exclude_dirs=None):
        """
        Perform the exponentiation operation for node v
        
        Args:
            v: Node ID
            exclude_dirs: Set of directions to exclude
            
        Returns:
            Number of new nodes learned
        """
        node = self.nodes[v]
        exclude_dirs = exclude_dirs or set()
        old_sv_size = len(node['sv'])
        
        # For each neighbor not in exclude_dirs
        for u in node['neighbors'] - exclude_dirs:
            # For each node w in Sv→u
            for w in list(node['sv']):
                if w == v:
                    continue
                
                rv_w = self.get_rv_mapping(v, w)
                if rv_w == u:  # If w is in the direction of u
                    # Get Sw ̸→rw(v)
                    rw_v = self.get_rv_mapping(w, v)
                    if rw_v is not None:
                        w_away = self.get_opposite_direction(w, rw_v) - {w}
                        node['sv'].update(w_away)
        
        # Ensure symmetric view
        for w in list(node['sv']):
            if w != v and v not in self.nodes[w]['sv']:
                self.nodes[w]['sv'].add(v)
        
        return len(node['sv']) - old_sv_size
    
    def compress_light_subtrees(self):
        """
        Execute the CompressLightSubTrees procedure
        
        Returns:
            New graph with light subtrees compressed
        """
        # Repeat for O(log D_hat) iterations
        iterations = int(math.log2(self.d_hat)) + 2
        print(f"Running CompressLightSubTrees for {iterations} iterations...")
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Process nodes in parallel (simulated)
            for v, node in self.nodes.items():
                if node['state'] != "active":
                    continue
                
                # Probe directions
                fullDirs, largestDir = self.probe_directions(v)
                
                # Determine which directions to exponentiate
                if len(fullDirs) >= 2:
                    # If multiple directions are too large, node is heavy
                    node['state'] = "sad"
                    
                elif len(fullDirs) == 1:
                    # Exponentiate in all directions except fullDirs
                    self.exponentiate(v, exclude_dirs=fullDirs)
                    
                elif largestDir is not None:
                    # Exponentiate in all directions except largestDir
                    self.exponentiate(v, exclude_dirs={largestDir})
                
                # Check if node becomes happy
                for u in node['neighbors']:
                    is_light, _ = self.is_light(v, u)
                    if is_light:
                        opposite = self.get_opposite_direction(v, u)
                        # Check if the subtree is fully contained in the node's view
                        if opposite.issubset(node['sv']):
                            node['state'] = "happy"
                            break
                
                # Check if node becomes full
                if len(node['sv']) >= 2 * self.n ** (self.delta/4):
                    node['state'] = "full"
            
            # Update max_id values
            self._propagate_max_ids()
            
            # Print statistics
            states = {"active": 0, "happy": 0, "sad": 0, "full": 0}
            for node in self.nodes.values():
                states[node['state']] += 1
            print(f"  States: {states}")
            
            # Check termination condition
            if states["active"] == 0:
                print("All nodes have decided their state, terminating early.")
                break
        
        # Compress light subtrees into heavy nodes
        G_prime = self.tree.copy()
        nodes_to_remove = []
        
        # Identify light nodes with at least one unhappy neighbor
        for v, node in self.nodes.items():
            if node['state'] == "happy":
                # Find an unhappy neighbor
                for u in node['neighbors']:
                    if self.nodes[u]['state'] != "happy":
                        # Compress v into u
                        for w in node['neighbors'] - {u}:
                            if w in G_prime.nodes and u in G_prime.nodes:
                                G_prime.add_edge(u, w)
                        nodes_to_remove.append(v)
                        break
        
        # Remove the compressed nodes
        for v in nodes_to_remove:
            if v in G_prime.nodes:
                G_prime.remove_node(v)
        
        # If all nodes are happy, compress into the highest ID node
        if all(node['state'] == "happy" for node in self.nodes.values()):
            max_id_node = max(self.tree.nodes())
            G_prime = nx.Graph()
            G_prime.add_node(max_id_node)
        
        return G_prime
    
    def compress_paths(self, G):
        """
        Execute the CompressPaths procedure
        
        Args:
            G: Graph from CompressLightSubTrees
            
        Returns:
            New graph with paths compressed
        """
        G_prime = G.copy()
        
        # Find degree-2 nodes that form paths
        path_nodes = [node for node, degree in G.degree() if degree == 2]
        
        while path_nodes:
            v = path_nodes[0]
            
            # Skip if node is no longer in the graph or no longer has degree 2
            if v not in G_prime.nodes() or G_prime.degree(v) != 2:
                path_nodes.pop(0)
                continue
            
            # Get the endpoints of the path segment
            neighbors = list(G_prime.neighbors(v))
            u, w = neighbors[0], neighbors[1]
            
            # Remove the node and add an edge between its neighbors
            G_prime.remove_node(v)
            G_prime.add_edge(u, w)
            
            # Update path_nodes
            path_nodes.pop(0)
            
            # Check if u or w now have degree 2 and add them to the list
            if u in G_prime.nodes() and G_prime.degree(u) == 2 and u not in path_nodes:
                path_nodes.append(u)
            if w in G_prime.nodes() and G_prime.degree(w) == 2 and w not in path_nodes:
                path_nodes.append(w)
        
        return G_prime
    
    def _propagate_max_ids(self):
        """
        Propagate maximum IDs across the tree
        Each node updates its max_id based on its knowledge set
        """
        changed = True
        while changed:
            changed = False
            for v, node in self.nodes.items():
                old_max = node['max_id']
                
                # Update max_id based on all nodes in view
                for w in node['sv']:
                    if self.nodes[w]['max_id'] > node['max_id']:
                        node['max_id'] = self.nodes[w]['max_id']
                        changed = True
    
    def solve(self):
        """
        Solve the MAX-ID problem for the tree
        
        Returns:
            Maximum ID in the tree
        """
        # Initialize phases
        phases = 0
        G = [self.tree.copy()]
        
        # Main algorithm loop - continue until we reach a single node
        while len(G[-1].nodes) > 1:
            print(f"\nPhase {phases}, nodes: {len(G[-1].nodes)}")
            
            # Step 1: CompressLightSubTrees
            G_prime = self.compress_light_subtrees()
            print(f"After CompressLightSubTrees: {len(G_prime.nodes)} nodes")
            
            # Step 2: CompressPaths
            G_next = self.compress_paths(G_prime)
            print(f"After CompressPaths: {len(G_next.nodes)} nodes")
            
            G.append(G_next)
            phases += 1
            
            # Propagate max IDs
            self._propagate_max_ids()
            
            if phases > 10:  # Safety limit
                print("Warning: Max phases reached")
                break
            
            if len(G_next.nodes) == 1:
                print("Graph reduced to a single node")
                break
        
        # Final propagation of max IDs
        self._propagate_max_ids()
        
        # Return the result (maximum ID found in the tree)
        return max(node['max_id'] for node in self.nodes.values())

# Create test cases
def create_test_cases():
    # Case 1: Path graph - challenging for exponentiation
    path_graph = nx.path_graph(10)
    # Relabel nodes with random IDs to make it interesting
    mapping = {i: i*10+5 for i in range(10)}
    path_graph = nx.relabel_nodes(path_graph, mapping)
    
    # Case 2: Balanced tree - easier for exponentiation
    balanced_tree = nx.balanced_tree(2, 3)  # Binary tree of depth 3
    # Relabel nodes
    mapping = {i: i*10+7 for i in range(len(balanced_tree.nodes()))}
    balanced_tree = nx.relabel_nodes(balanced_tree, mapping)
    
    # Case 3: Star-like tree with branches - mix of light and heavy nodes
    mixed_tree = nx.Graph()
    mixed_tree.add_node(100)  # Center node with high ID
    # Add some branches
    for i in range(1, 6):
        mixed_tree.add_edge(100, 200+i)
        if i <= 3:  # Add some additional branches
            mixed_tree.add_edge(200+i, 300+i)
            mixed_tree.add_edge(300+i, 400+i)
    
    return path_graph, balanced_tree, mixed_tree

# Helper function to visualize a tree
def visualize_tree(tree, node_colors=None, title="Tree Visualization"):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(tree, seed=42)
    
    # Default node colors
    if node_colors is None:
        node_colors = "lightblue"
    
    nx.draw(tree, pos, with_labels=True, node_color=node_colors, 
            node_size=700, font_size=10, font_weight='bold', edge_color='gray')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Test our implementation
def run_tests():
    path_graph, balanced_tree, mixed_tree = create_test_cases()
    
    # Case 1: Path Graph
    print("\n===== CASE 1: PATH GRAPH =====")
    visualize_tree(path_graph, title="Path Graph - Before")
    solver = MAXIDSolver(path_graph, d_hat=10)
    max_id = solver.solve()
    print(f"Maximum ID found: {max_id}")
    
    # Visualize the final state
    node_colors = []
    for node_id in path_graph.nodes():
        if solver.nodes[node_id]['state'] == "happy":
            node_colors.append("lightgreen")
        elif solver.nodes[node_id]['state'] == "sad":
            node_colors.append("salmon")
        elif solver.nodes[node_id]['state'] == "full":
            node_colors.append("purple")
        else:
            node_colors.append("lightblue")
    
    visualize_tree(path_graph, node_colors=node_colors, title="Path Graph - After")
    
    # Case 2: Balanced Tree
    print("\n===== CASE 2: BALANCED TREE =====")
    visualize_tree(balanced_tree, title="Balanced Tree - Before")
    solver = MAXIDSolver(balanced_tree, d_hat=6)
    max_id = solver.solve()
    print(f"Maximum ID found: {max_id}")
    
    # Visualize the final state
    node_colors = []
    for node_id in balanced_tree.nodes():
        if solver.nodes[node_id]['state'] == "happy":
            node_colors.append("lightgreen")
        elif solver.nodes[node_id]['state'] == "sad":
            node_colors.append("salmon")
        elif solver.nodes[node_id]['state'] == "full":
            node_colors.append("purple")
        else:
            node_colors.append("lightblue")
    
    visualize_tree(balanced_tree, node_colors=node_colors, title="Balanced Tree - After")
    
    # Case 3: Mixed Tree (Star with Branches)
    print("\n===== CASE 3: MIXED TREE (STAR WITH BRANCHES) =====")
    visualize_tree(mixed_tree, title="Mixed Tree - Before")
    solver = MAXIDSolver(mixed_tree, d_hat=6)
    max_id = solver.solve()
    print(f"Maximum ID found: {max_id}")
    
    # Visualize the final state
    node_colors = []
    for node_id in mixed_tree.nodes():
        if solver.nodes[node_id]['state'] == "happy":
            node_colors.append("lightgreen")
        elif solver.nodes[node_id]['state'] == "sad":
            node_colors.append("salmon")
        elif solver.nodes[node_id]['state'] == "full":
            node_colors.append("purple")
        else:
            node_colors.append("lightblue")
    
    visualize_tree(mixed_tree, node_colors=node_colors, title="Mixed Tree - After")

if __name__ == "__main__":
    run_tests()
