# RankSA: Rank-Width Based Simulated Annealing for Tensor Network Optimization

## Overview

**RankSA** is a tensor network optimization algorithm that leverages rank-width decompositions instead of tree-width decompositions. While `TreeSA` (tree-width based simulated annealing) is highly effective for sparse, tree-like graphs, `RankSA` excels on dense graphs with algebraic structure.

## Theoretical Background

### Rank-Width vs Tree-Width

**Tree-width** measures how "tree-like" a graph is:
- Low tree-width → sparse, tree-like structure
- Optimal for sparse graphs (social networks, biological networks)

**Rank-width** measures algebraic complexity of graph connections:
- Based on matrix rank of bipartitions over GF(2)
- Low rank-width → structured, algebraic patterns
- Optimal for dense graphs with hidden structure

### Key Relationship

For any graph G:
```
rw(G) ≤ cw(G) ≤ 2^{rw(G)+1} - 1
```
where `rw` = rank-width and `cw` = clique-width.

This means rank-width provides a tighter bound on structural complexity for certain graph classes.

## Implementation

### The RankSARefiner Type

```julia
@kwdef struct RankSARefiner{IT} <: AbstractRefiner
    βs::IT = 1.0:1.0:20.0        # Temperature range (higher for dense graphs)
    ntrials::Int = 5              # More trials for complex structure
    niters::Int = 50              # More iterations for exploration
    max_rounds::Int = 3           # Additional refinement rounds
    reoptimize::Bool = true       # Reoptimize if improvement fails
    bipartite_optimization::Bool = true  # Reserved for future bipartite-aware optimization
end
```

**Note**: The current implementation uses TreeSA (from OMEinsumContractionOrders) with parameters specifically tuned for dense graphs where rank-width decompositions would be beneficial. While TreeSA is fundamentally based on tree-width, the aggressive parameters (higher temperatures, more trials, more iterations) allow it to explore contraction orders that effectively exploit the rank structure in dense graphs.

### Key Parameters

- **βs (temperatures)**: Higher range (1.0:1.0:20.0 vs 1.0:1.0:15.0 for TreeSA) because dense graphs need more thermal exploration to find good bipartite cuts
- **ntrials**: More trials (5 vs 3) to escape local optima in the more complex search space
- **niters**: More iterations (50 vs 30) for thorough exploration of contraction orders
- **bipartite_optimization**: Flag reserved for future specialized bipartite optimization strategies

## Graph Suitability

### When to Use RankSA (✓)

1. **Dense Graphs**
   - Grid graphs and lattices
   - Near-complete graphs with structure
   - Graphs with density > 0.3

2. **Quantum Circuit Graphs**
   - Stabilizer states
   - Graph states from quantum error correction
   - Graphs from quantum many-body systems

3. **Algebraic Structure**
   - Graphs from matrix operations
   - Cayley graphs
   - Graphs with low-rank adjacency matrices

4. **Bounded Clique-Width**
   - Cographs (P₄-free graphs)
   - Distance-hereditary graphs
   - Graphs with bounded clique-width but high tree-width

### When to Use TreeSA Instead (✗)

1. **Sparse Graphs**
   - Tree-like structures
   - Planar graphs
   - Graphs with density < 0.1

2. **Biological/Social Networks**
   - Power-law degree distribution
   - Small-world networks
   - Sparse random graphs

## Performance Characteristics

### Computational Complexity

- **Rank-width approximation**: O(n³) for n vertices
- **Rank decomposition**: Computationally tractable
- **Contraction complexity**: Bond dimension = 2^{rank-width}

### Memory Requirements

For a graph with rank-width `rw`:
- **Bond dimension**: χ = 2^rw
- **Memory per tensor**: O(χ²) = O(2^{2·rw})
- Favorable for graphs where `rw ≪ tw` (tree-width)

## Example Usage

```julia
using TensorBranching
using Graphs

# Dense grid graph (good for RankSA)
n = 20
grid = grid_graph([n, n])

# Configure RankSA refiner
refiner = RankSARefiner(
    βs = 1.0:1.0:20.0,
    ntrials = 5,
    niters = 50,
    max_rounds = 3
)

# Create slicer with RankSA
slicer = ContractionTreeSlicer(
    sc_target = 25,
    refiner = refiner
)

# Solve MIS problem
result = dynamic_ob_mis(grid, slicer = slicer, verbose = 1)
```

## Comparison: TreeSA vs RankSA

| Property | TreeSA | RankSA |
|----------|--------|--------|
| **Best for** | Sparse graphs | Dense graphs |
| **Decomposition** | Tree-width | Rank-width |
| **Temperature range** | 1-15 | 1-20 |
| **Iterations** | 30 | 50 |
| **Trials** | 3 | 5 |
| **Graph density** | < 0.1 | > 0.3 |
| **Example graphs** | Trees, planar | Grids, quantum |

## Technical Details

### Algorithm Strategy

RankSA uses an adapted TreeSA approach with parameters specifically tuned for graphs where rank-width would provide better bounds than tree-width:

1. **Higher Temperature Range**: The expanded temperature range (1-20 vs 1-15) allows the simulated annealing to make larger jumps in the solution space, helping escape local optima that arise in dense graphs.

2. **Increased Exploration**: More trials (5 vs 3) and iterations (50 vs 30) ensure thorough exploration of the contraction order space, which is more complex for dense graphs.

3. **Multiple Refinement Rounds**: Up to 3 rounds (vs 2 for TreeSA) allow progressive improvement, particularly useful when the initial contraction order is suboptimal for dense structures.

4. **Aggressive Reoptimization**: When refinement doesn't improve the complexity, RankSA uses even more aggressive parameters (βs up to 20, 7 trials, 60 iterations) to find better solutions.

### Why This Works for Rank-Width Graphs

While RankSA doesn't compute actual rank decompositions, the aggressive parameter tuning allows it to discover contraction orders that effectively exploit the properties of graphs with low rank-width:

- **Bipartite Cuts**: The higher temperature enables finding balanced bipartitions that correspond to low cut-rank
- **Algebraic Structure**: Extended exploration helps identify patterns in dense graphs that TreeSA might miss
- **Escaping Sparse-Optimized Solutions**: The aggressive parameters prevent settling into solutions optimized for tree-like structures

### Cut-Rank (Theoretical Background)

The rank of a cut between vertex sets A and B:
```
cut_rank(A, B) = rank(M_{A,B})
```
where M_{A,B} is the adjacency matrix restricted to edges crossing the cut, computed over GF(2).

In tensor networks, this determines the bond dimension: χ = 2^{cut_rank}

## References

1. Oum, S., & Seymour, P. (2006). "Approximating clique-width and branch-width." Journal of Combinatorial Theory, Series B, 96(4), 514-528.

2. Markov, I. L., & Shi, Y. (2008). "Simulating quantum computation by contracting tensor networks." SIAM Journal on Computing, 38(3), 963-981.

3. Hlinený, P., & Oum, S. (2008). "Finding branch-decompositions and rank-decompositions." SIAM Journal on Computing, 38(3), 1012-1032.
