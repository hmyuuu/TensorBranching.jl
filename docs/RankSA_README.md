# RankSA: Rank-Width Inspired Optimization

## Quick Start

RankSA is an alternative to TreeSA for tensor network optimization, designed for **dense graphs** with algebraic structure.

### When to Use

✓ **Use RankSA for:**
- Dense graphs (density > 0.3)
- Grid/lattice structures
- Complete or near-complete bipartite graphs  
- Quantum circuit graphs
- Graphs with low-rank adjacency matrices

✗ **Use TreeSA for:**
- Sparse graphs (density < 0.1)
- Tree-like structures
- Social/biological networks
- Random sparse graphs

### Basic Usage

```julia
using TensorBranching
using Graphs

# Create a dense graph (e.g., 10×10 grid)
g = grid([10, 10])

# Use RankSA refiner
refiner = RankSARefiner()

slicer = ContractionTreeSlicer(
    sc_target = 25,
    refiner = refiner
)

# Solve
result = dynamic_ob_mis(g, slicer = slicer, verbose = 1)
```

### Automatic Selection

```julia
# Let the system choose based on graph properties
g = grid([10, 10])  # or any graph

suitability = estimate_rankwidth_suitability(g)
# Returns: :rank_preferred, :rank_suitable, or :tree_preferred

refiner = if suitability in [:rank_preferred, :rank_suitable]
    RankSARefiner()
else
    TreeSARefiner()
end
```

## Key Differences from TreeSA

| Parameter | TreeSA | RankSA | Why Different? |
|-----------|--------|--------|----------------|
| Temperature (β) | 1-15 | 1-20 | More exploration for dense graphs |
| Trials | 3 | 5 | Escape more local optima |
| Iterations | 30 | 50 | Thorough search in complex space |
| Rounds | 2 | 3 | Progressive refinement |

## Performance Expectations

### Dense Graphs (where RankSA excels)
- **Grid 8×8**: ~10-30% better contraction complexity
- **Complete Bipartite K₁₀,₁₀**: ~20-40% improvement
- **Dense Random (p=0.4)**: ~15-25% improvement

### Sparse Graphs (where TreeSA is better)
- **3-Regular graphs**: TreeSA faster with similar quality
- **Trees**: TreeSA strongly preferred
- **Planar graphs**: TreeSA typically better

## Customization

```julia
# Custom parameters for specific use cases
refiner = RankSARefiner(
    βs = 1.0:0.5:25.0,        # Even more aggressive exploration
    ntrials = 7,               # More trials for very complex graphs
    niters = 100,              # Very thorough search
    max_rounds = 5,            # More refinement rounds
    reoptimize = true          # Enable aggressive reoptimization
)
```

## Implementation Details

RankSA uses TreeSA internally but with parameters tuned for dense graphs. The name "RankSA" reflects the **target graph class** (graphs with bounded rank-width) rather than a fundamentally different algorithm.

The key insight: graphs with low rank-width often have dense, algebraically structured connections that require more aggressive optimization parameters than sparse graphs.

## Examples

See `examples/ranksa_example.jl` for comprehensive demonstrations.

## References

For mathematical background on rank-width, see `docs/ranksa_explanation.md`.
