using TensorBranching
using Graphs, OMEinsum
using OptimalBranching
using OptimalBranching.OptimalBranchingCore, OptimalBranching.OptimalBranchingMIS
using GenericTensorNetworks
using Test
using Random

Random.seed!(1234)

@testset "RankSARefiner basic functionality" begin
    @testset "RankSARefiner construction" begin
        refiner = RankSARefiner()
        @test refiner.βs == 1.0:1.0:20.0
        @test refiner.ntrials == 5
        @test refiner.niters == 50
        @test refiner.max_rounds == 3
        @test refiner.reoptimize == true
        @test refiner.bipartite_optimization == true
    end
    
    @testset "Custom RankSARefiner parameters" begin
        custom_refiner = RankSARefiner(
            βs = 1.0:2.0:25.0,
            ntrials = 7,
            niters = 60,
            max_rounds = 4,
            reoptimize = false,
            bipartite_optimization = false
        )
        @test custom_refiner.βs == 1.0:2.0:25.0
        @test custom_refiner.ntrials == 7
        @test custom_refiner.niters == 60
        @test custom_refiner.max_rounds == 4
        @test custom_refiner.reoptimize == false
        @test custom_refiner.bipartite_optimization == false
    end
end

@testset "Graph density utilities" begin
    @testset "graph_density calculation" begin
        g_complete = complete_graph(10)
        @test graph_density(g_complete) ≈ 1.0
        
        g_empty = SimpleGraph(10)
        @test graph_density(g_empty) ≈ 0.0
        
        g_star = star_graph(10)
        expected_density = 2 * 9 / (10 * 9)
        @test graph_density(g_star) ≈ expected_density
    end
    
    @testset "estimate_rankwidth_suitability" begin
        dense_graph = erdos_renyi(50, 0.5)
        @test estimate_rankwidth_suitability(dense_graph) == :rank_preferred
        
        medium_graph = erdos_renyi(60, 0.2)
        
        sparse_graph = random_regular_graph(50, 3)
        @test estimate_rankwidth_suitability(sparse_graph) == :tree_preferred
    end
end

@testset "RankSA refinement on small graphs" begin
    for seed in 1:3
        Random.seed!(seed)
        
        g = grid([6, 6])
        net = GenericTensorNetwork(IndependentSet(g), optimizer=TreeSA())
        code = true_eincode(net.code)
        size_dict = uniformsize(code, 2)
        
        refiner = RankSARefiner(max_rounds = 1, ntrials = 2, niters = 10)
        sc0 = contraction_complexity(code, size_dict).sc
        
        refined_code = refine(code, size_dict, refiner, 25, sc0)
        
        @test !isnothing(refined_code)
        @test refined_code isa DynamicNestedEinsum
        
        tensors = GenericTensorNetworks.generate_tensors(TropicalF32(1.0), net)
        result_original = code(tensors...)[].n
        result_refined = refined_code(tensors...)[].n
        
        @test result_original ≈ result_refined
    end
end

@testset "RankSA vs TreeSA on different graph types" begin
    @testset "Dense grid graph" begin
        g = grid([8, 8])
        
        tree_refiner = TreeSARefiner(max_rounds = 1, ntrials = 2, niters = 10)
        rank_refiner = RankSARefiner(max_rounds = 1, ntrials = 2, niters = 10)
        
        net = GenericTensorNetwork(IndependentSet(g), optimizer=TreeSA())
        code = true_eincode(net.code)
        size_dict = uniformsize(code, 2)
        sc0 = contraction_complexity(code, size_dict).sc
        
        tree_refined = refine(deepcopy(code), size_dict, tree_refiner, 25, sc0)
        rank_refined = refine(deepcopy(code), size_dict, rank_refiner, 25, sc0)
        
        tensors = GenericTensorNetworks.generate_tensors(TropicalF32(1.0), net)
        @test tree_refined(tensors...)[].n ≈ rank_refined(tensors...)[].n
    end
    
    @testset "Sparse regular graph" begin
        g = random_regular_graph(30, 3)
        
        tree_refiner = TreeSARefiner(max_rounds = 1, ntrials = 2, niters = 10)
        rank_refiner = RankSARefiner(max_rounds = 1, ntrials = 2, niters = 10)
        
        net = GenericTensorNetwork(IndependentSet(g), optimizer=TreeSA())
        code = true_eincode(net.code)
        size_dict = uniformsize(code, 2)
        sc0 = contraction_complexity(code, size_dict).sc
        
        tree_refined = refine(deepcopy(code), size_dict, tree_refiner, 25, sc0)
        rank_refined = refine(deepcopy(code), size_dict, rank_refiner, 25, sc0)
        
        tensors = GenericTensorNetworks.generate_tensors(TropicalF32(1.0), net)
        @test tree_refined(tensors...)[].n ≈ rank_refined(tensors...)[].n
    end
end

@testset "RankSA integration with dynamic_ob_mis" begin
    for seed in 1:2
        Random.seed!(seed)
        
        g = grid([5, 5])
        
        rank_slicer = ContractionTreeSlicer(
            sc_target = 20,
            refiner = RankSARefiner(max_rounds = 1, ntrials = 1, niters = 5)
        )
        
        result = dynamic_ob_mis(g, slicer = rank_slicer, verbose = 0)
        
        @test result isa Number
        @test result > 0
    end
end

@testset "Bipartite optimization detection" begin
    @testset "Complete bipartite graph" begin
        g = complete_bipartite_graph(8, 8)
        @test is_bipartite(g)
        @test graph_density(g) > 0.3
        
        net = GenericTensorNetwork(IndependentSet(g), optimizer=TreeSA())
        code = true_eincode(net.code)
        size_dict = uniformsize(code, 2)
        
        refiner = RankSARefiner(
            bipartite_optimization = true,
            max_rounds = 1,
            ntrials = 1,
            niters = 5
        )
        
        refined = refine(code, size_dict, refiner, 20, 100.0)
        @test !isnothing(refined)
    end
end
