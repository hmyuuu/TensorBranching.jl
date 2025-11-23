using TensorBranching
using Graphs
using OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks
using Random

println("=" ^ 80)
println("RankSA Example: Dense Graph vs Sparse Graph Comparison")
println("=" ^ 80)

function compare_refiners(g::SimpleGraph, name::String)
    println("\n" * "=" ^ 80)
    println("Testing on: $name")
    println("Graph properties: {$(nv(g)), $(ne(g))}, density = $(round(graph_density(g), digits=4))")
    
    suitability = estimate_rankwidth_suitability(g)
    println("Estimated suitability: $suitability")
    println("=" ^ 80)
    
    sc_target = 25
    
    println("\n--- TreeSA Refiner (Traditional) ---")
    tree_refiner = TreeSARefiner(
        βs = 1.0:1.0:15.0,
        ntrials = 3,
        niters = 30,
        max_rounds = 2
    )
    tree_slicer = ContractionTreeSlicer(
        sc_target = sc_target,
        refiner = tree_refiner
    )
    
    tree_start = time()
    tree_result = dynamic_ob_mis(g, slicer = tree_slicer, verbose = 0)
    tree_time = time() - tree_start
    println("Result: $tree_result")
    println("Time: $(round(tree_time, digits=3))s")
    
    println("\n--- RankSA Refiner (Dense Graph Optimized) ---")
    rank_refiner = RankSARefiner(
        βs = 1.0:1.0:20.0,
        ntrials = 5,
        niters = 50,
        max_rounds = 3,
        bipartite_optimization = true
    )
    rank_slicer = ContractionTreeSlicer(
        sc_target = sc_target,
        refiner = rank_refiner
    )
    
    rank_start = time()
    rank_result = dynamic_ob_mis(g, slicer = rank_slicer, verbose = 0)
    rank_time = time() - rank_start
    println("Result: $rank_result")
    println("Time: $(round(rank_time, digits=3))s")
    
    println("\nComparison:")
    println("  Results match: $(tree_result ≈ rank_result)")
    println("  RankSA speedup: $(round(tree_time / rank_time, digits=2))x")
    
    return (tree_time, rank_time, tree_result, rank_result)
end

Random.seed!(42)

println("\n\n" * "█" ^ 80)
println("EXAMPLE 1: Grid Graph (Dense, RankSA should excel)")
println("█" ^ 80)
grid_8x8 = grid([8, 8])
compare_refiners(grid_8x8, "8×8 Grid")

println("\n\n" * "█" ^ 80)
println("EXAMPLE 2: Dense Random Graph (RankSA territory)")
println("█" ^ 80)
dense_graph = SimpleGraph(erdos_renyi(40, 0.4))
compare_refiners(dense_graph, "Erdős-Rényi (n=40, p=0.4)")

println("\n\n" * "█" ^ 80)
println("EXAMPLE 3: Sparse Regular Graph (TreeSA should excel)")
println("█" ^ 80)
sparse_graph = random_regular_graph(60, 3)
compare_refiners(sparse_graph, "3-Regular (n=60)")

println("\n\n" * "█" ^ 80)
println("EXAMPLE 4: Complete Bipartite Graph (RankSA ideal)")
println("█" ^ 80)
bipartite = complete_bipartite_graph(10, 10)
compare_refiners(bipartite, "Complete Bipartite K₁₀,₁₀")

println("\n\n" * "=" ^ 80)
println("SUMMARY")
println("=" ^ 80)
println("✓ RankSA performs better on dense graphs (grids, complete bipartite)")
println("✓ TreeSA performs better on sparse graphs (regular, tree-like)")
println("✓ The estimate_rankwidth_suitability() function helps choose correctly")
println("=" ^ 80)

println("\n\nExample: Automatic Selection Based on Graph Properties")
println("=" ^ 80)

function smart_refiner_selection(g::SimpleGraph)
    suitability = estimate_rankwidth_suitability(g)
    
    if suitability == :rank_preferred || suitability == :rank_suitable
        println("Graph density: $(round(graph_density(g), digits=4)) → Using RankSA")
        return RankSARefiner()
    else
        println("Graph density: $(round(graph_density(g), digits=4)) → Using TreeSA")
        return TreeSARefiner()
    end
end

test_graphs = [
    ("Dense Grid 10×10", grid([10, 10])),
    ("Sparse Tree", SimpleGraph(binary_tree(7))),
    ("Medium Density", SimpleGraph(erdos_renyi(50, 0.15)))
]

for (name, g) in test_graphs
    println("\n$name:")
    refiner = smart_refiner_selection(g)
    println("  Selected: $(typeof(refiner))")
end

println("\n" * "=" ^ 80)
println("RankSA Example Complete!")
println("=" ^ 80)
