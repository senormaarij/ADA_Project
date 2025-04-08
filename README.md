# ADA_Project: Optimal Deterministic Massively Parallel Connectivity on Forests Implementation

## Project Overview

This project aims to implement the algorithm proposed in the paper Optimal Deterministic Massively Parallel Connectivity on Forests. The goal is to identify connected components in massive graphs within the Massively Parallel Computation (MPC) model. Since no official implementation is available, we will develop it manually using Python and Dask, with potential GPU acceleration for handling large datasets/graphs.

## Current Progress

- The theoretical framework and algorithm flow have been thoroughly studied.
- Section 4 of the paper (Data Structures and Techniques) has been carefully summarized and analyzed.
- We are in the process of implementing individual components as modular, testable functions.
- Small-scale unit tests for specific compression routines are being built and evaluated.

## Known Limitations (Current Stage)

- The original algorithm is tailored to the theoretical MPC model, which assumes perfect load balancing and synchronous rounds. Our Dask-based setup approximates this but may introduce real-world variance.
- Graph data distribution across machines is being handled manually using Dask, rather than relying on a cluster-aware environment.
- GPU acceleration remains experimental, and integration with RAPIDS is planned only after the CPU version stabilizes.
- The MAX-ID step, assumed to be solved in a single machine in the paper, will need fallback strategies when graph size doesn't permit this.
- Compression phase correctness is still being verified through synthetic test graphs.

## Planned Implementation Workflow

### Language and Libraries

- Language: Python
- Parallel Framework: Dask (local distributed scheduler)
- Optional Acceleration: CuPy, RAPIDS

### Data

- We are generating synthetic forests of various sizes and depths for testing compression and connectivity logic.
- Graphs are stored and manipulated using distributed adjacency lists and sparse representations.

### Step-by-Step Implementation Plan

1. Implement basic graph representation in Dask arrays and dictionaries.
2. `CompressLightSubTrees`: Merge light subtrees based on \(n^{\delta/8}\) thresholds.
3. `CompressPaths`: Path compression logic for degree-2 node chains.
4. `MAX-ID` Simulation: Use local computation for final decision-making.
5. Decompression Phase: Restore results to original nodes.
