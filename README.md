# Project Overview

This project implements the MAX-ID algorithm as a key component for solving the Connected Components Problem on forests (tree-structured graphs) in a simplified computational model. The algorithm is inspired by the paper "Optimal Deterministic Massively Parallel Connectivity on Forests." Our goal is to compute the maximum node ID for each tree in a forest and use this to identify connected components, where each node outputs the maximum ID of its tree to indicate its component.

Since the original algorithm is designed for the theoretical Massively Parallel Computation (MPC) model, which is complex to implement practically, we are developing a single-threaded Python simulation to test and validate the algorithm's logic on synthetic graphs.

The MAX-ID problem involves assigning each node in a tree the maximum node ID within its connected component (tree). By solving MAX-ID, we can solve the Connected Components Problem: nodes with the same maximum ID belong to the same tree. This project focuses on the MAX-ID solver and its application to connected components, ignoring the Locally Checkable Labeling (LCL) aspects of the paper, as they are too complex for our scope.

## Current Progress

The theoretical framework of the MAX-ID algorithm and its role in solving the Connected Components Problem have been thoroughly studied.

Section 4 of the paper (Data Structures and Techniques) has been summarized, focusing on the CompressLightSubTrees and CompressPaths phases.

Modular, testable functions for the MAX-ID solver are being implemented, including:

Graph representation using adjacency lists.

CompressLightSubTrees to merge light subtrees based on n^{delta/8} thresholds.

CompressPaths to compress chains of degree-2 nodes.

Small-scale unit tests are being developed for compression routines using synthetic tree graphs.


## Known Limitations

The original algorithm assumes the MPC model, which requires independent machines with synchronized rounds and no shared memory. Implementing a true MPC model is infeasible on a single machine or cluster (no GPU or multi-threading is used), so we simulate the algorithm sequentially in Python.

The MAX-ID solver is designed for trees, not general graphs, limiting its applicability to forests.

Compression phase correctness is still being verified through synthetic test graphs (e.g., paths and stars).

Large-scale testing is pending until the core algorithm is stable.

The decompression phase (to propagate results back to all nodes) is simplified, as our simulation outputs the maximum ID for the remaining node(s).

## How the MAX-ID Solver Works?

The MAX-ID algorithm computes the maximum node ID for each tree in a forest, enabling nodes to identify their connected component. Here's how it works:

### Initialization

Each node has a unique ID and a knowledge set (S_v), initially its neighbors.

Nodes track a max_id, initially their own ID.

### Compression Phases

CompressLightSubTrees

Nodes estimate subtree sizes and classify as:

Happy: Has a "light" subtree (<= n^{delta/8} nodes).

Full: Knowledge set exceeds memory limit (n^{delta}).

Sad: Has multiple heavy subtrees.

Happy nodes compress into full or sad neighbors, transferring their max_id and updating the graph.

CompressPaths

Chains of degree-2 nodes are compressed into a single edge between their endpoints.

The maximum max_id of the path is assigned to the endpoints.

These phases reduce the tree's size while propagating the maximum ID.

### Termination

The tree is reduced to a single node, whose max_id is the maximum ID of the tree.

All nodes in the same tree (connected component) would output this max_id.

Solving the Connected Components Problem

## The Connected Components Problem requires labeling each node in a graph with an identifier for its connected component. For forests:

Each tree is a connected component.

The MAX-ID algorithm assigns the maximum node ID of each tree to all its nodes.

Nodes with the same max_id are in the same tree, solving the problem.

Example

Tree: 1 -- 2 -- 3 -- 4 -- 5
After compression (e.g., node 4 into 3, then 5 into 3, etc.), one node remains with max_id = 5. All nodes output 5, indicating they belong to the same connected component.

### Efficiency

Compression Reduces Size: The tree is reduced to a single node, minimizing computation.

Maximum ID as Label: The unique maximum ID serves as a consistent component identifier.

Scalability: In the theoretical MPC model, the algorithm runs in O(log D) rounds (D is the diameter), suitable for large graphs.

In our simulation, we output the max_id for the remaining node(s), simulating the result for all nodes in each tree.

## Why Simplify the MPC Model?

The MPC model assumes:

Independent machines with local memory (O(n^{delta})).

Synchronous communication rounds.

No shared memory or multi-threading.

Implementing this on a single machine or cluster is impractical because:

Independent Machines: A single machine cannot replicate isolated MPC machines without complex virtualization.

No GPU/Multi-threading: The MPC model prohibits shared-memory parallelism, ruling out GPU or multi-threaded optimizations.

Synchronous Rounds: Simulating synchronized rounds on a cluster (e.g., with Dask) introduces overhead and variance.

Instead, we use a single-threaded Python simulation to:

Validate the algorithm's logic.

Test compression and MAX-ID computation on small graphs.

Avoid the complexity of distributed systems while preserving the algorithm's core functionality.

## Technical Details

Language: Python

Data Structures: Adjacency lists (dictionaries) and node objects for graph representation.

Testing: Unit tests with synthetic graphs (paths, stars, random trees).

## Data

Synthetic forests of various sizes (e.g., 5 to 100 nodes) and structures (paths, stars, balanced trees).

Graphs stored as dictionaries mapping node IDs to sets of neighbor IDs.

## Step-by-Step Implementation Plan

### Graph Representation:

Implement adjacency lists in Python dictionaries.

Store node attributes (ID, S_v, max_id, state) in a Node class.

### CompressLightSubTrees:

Merge light subtrees based on n^{delta/8} thresholds.

Update max_id and rewire the graph.

### CompressPaths:

Compress chains of degree-2 nodes into single edges.

Propagate maximum max_id to endpoints.

### MAX-ID Simulation:

Reduce each tree to a single node with the maximum ID.

### Testing and Validation:

Test on synthetic graphs to verify correctness.

## How to Run the Code

To run the MAX-ID solver implementation, follow these steps:

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Setup and Installation

1. Navigate to the project's source directory:
   ```bash
   cd src
   ```
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
### Running the code:
You can run the implementation using the following commands:

### MAX-ID Solver Simulation
This is the main implementation of the MAX-ID algorithm for connected components in forests:
```bash
python max-id-solver-simulation.py
```

### Parallel Simulation:
Run the parallel version of the implementation:

```bash
python parallel.py
```

### Example Usage:

The MAX-ID solver will run on the synthetic graphs defined in the code and output the maximum ID for each connected component. The output will show the progression of the algorithm through the compression phases and the final maximum ID assignment for each node.
