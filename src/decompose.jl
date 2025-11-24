# transform optimized eincode to elimination order
function eincode2order(code::NestedEinsum{L}) where {L}
    OMEinsum.isleaf(code) && return Vector{L}()
    
    elimination_order = Vector{L}()
    # Pre-allocate with a reasonable size hint to reduce allocations
    sizehint!(elimination_order, 32)
    
    for node in PostOrderDFS(code)
        (node isa LeafString) && continue
        eliminated = setdiff(vcat(getixsv(node.eins)...), getiyv(node.eins))
        append!(elimination_order, eliminated)
    end
    return reverse!(elimination_order)
end

function eincode2graph(code::Union{NestedEinsum, EinCode})
    fcode = code isa NestedEinsum ? flatten(code) : code
    indices = uniquelabels(fcode)
    (indices isa Vector{Int}) && sort!(indices)
    g = SimpleGraph(length(indices))
    id_dict = Dict(id => i for (i, id) in enumerate(indices))
    for xs in [getixsv(fcode); getiyv(fcode)]
        for i in 1:length(xs)-1
            for j in i+1:length(xs)
                add_edge!(g, id_dict[xs[i]], id_dict[xs[j]])
            end
        end
    end
    return g, id_dict
end

function map_tree_leaves(tree::Union{ContractionTree, Int}, mapping::Dict{Int, Int})
    return tree isa ContractionTree ? ContractionTree(map_tree_leaves(tree.left, mapping), map_tree_leaves(tree.right, mapping)) : mapping[tree]
end

function build_balanced_tree(leaves::Vector{Int})
    isempty(leaves) && error("empty leaf set for contraction tree")
    length(leaves) == 1 && return leaves[1]
    
    nodes = Vector{Union{ContractionTree, Int}}(leaves)
    # Pre-allocate next level to reduce allocations
    while length(nodes) > 1
        next_level_size = (length(nodes) + 1) ÷ 2
        next = Vector{Union{ContractionTree, Int}}()
        sizehint!(next, next_level_size)
        
        i = 1
        while i <= length(nodes)
            if i == length(nodes)
                push!(next, nodes[i])
                i += 1
            else
                push!(next, ContractionTree(nodes[i], nodes[i+1]))
                i += 2
            end
        end
        nodes = next
    end
    return nodes[1]
end

"""
    eo2ct(grouped_eo, incidence_list) -> ContractionTree

Construct a contraction tree from a grouped elimination order using the
current OMEinsumContractionOrders data structures.

Arguments
- `grouped_eo`: vector of vertex groups (each group is a vector of graph vertex ids)
- `incidence_list`: incidence structure mapping tensor indices to labels and
  providing `e2v::Dict{Int, Vector{Int}}` from vertex id to tensor indices

Returns
- `ContractionTree` combining all tensors associated with each group, then
  combining groups into a single binary tree

Example
```julia
rcode = rawcode(IndependentSet(g))
ixs = getixsv(rcode)
incidence_list = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]))
grouped_eo = [[i] for i in eo]
ct = eo2ct(grouped_eo, incidence_list)
code = decorate(parse_eincode(incidence_list, ct, vertices=collect(1:length(ixs))))
```
"""
function eo2ct(grouped_eo::Vector{<:AbstractVector{Int}}, incidence_list::IncidenceList)
    isempty(grouped_eo) && error("empty elimination order")
    
    # Pre-allocate trees vector with size hint
    trees = Vector{Union{ContractionTree, Int}}()
    sizehint!(trees, length(grouped_eo))
    
    for grp in grouped_eo
        # Directly use the group as leaves if it's already a Vector{Int}
        # Otherwise collect efficiently
        leaves = grp isa Vector{Int} ? grp : collect(grp)
        push!(trees, build_balanced_tree(leaves))
    end
    
    # Build final tree by reducing from left to right
    return isempty(trees) ? error("no trees to combine") : reduce((x, y) -> ContractionTree(x, y), trees)
end



function decompose(code::NestedEinsum{L}) where {L}
    g, id_dict = eincode2graph(code)
    labels = collect(keys(id_dict))[sortperm(collect(values(id_dict)))]
    return decomposition_tree(g, eincode2order(code), labels = labels)
end

function max_bag(tree::DecompositionTreeNode)
    max_bag_node = tree.bag
    max_size = length(max_bag_node)
    for node in PostOrderDFS(tree)
        bag_size = length(node.bag)
        if bag_size > max_size
            max_bag_node = node.bag
            max_size = bag_size
        end
    end
    return max_bag_node
end

# this function maps an elimination order on a old graph to a new graph with some vertices removed or reordered
function update_order(eo_old::Vector{Int}, vmap::Vector{Int})
    ivmap = inverse_vmap_dict(vmap)
    # Pre-allocate with size hint (at most length of eo_old)
    eo_new = Vector{Int}()
    sizehint!(eo_new, length(eo_old))
    
    for v in eo_old
        if haskey(ivmap, v)
            push!(eo_new, ivmap[v])
        end
    end
    return eo_new
end
function update_tree(g_new::SimpleGraph{Int}, eo_old::Vector{Int}, vmap::Vector{Int})
    eo_new = update_order(eo_old, vmap)
    return decomposition_tree(g_new, eo_new)
end

# reconstruct the contraction order from the grouped elimination order
# if set use_tree to true, the decomposition tree will be constructed to get a better elimination order
function order2eincode(g::SimpleGraph{Int}, eo::Vector{Int}; use_tree::Union{Bool,Symbol} = false)
    return GenericTensorNetwork(IndependentSet(g)).code
end

function update_code(g_new::SimpleGraph{Int}, code_old::NestedEinsum, vmap::Vector{Int})
    eo_old = eincode2order(code_old)
    eo_new = update_order(eo_old, vmap)
    return order2eincode(g_new, eo_new)
end

function _collect_tensorindices(code)
    if hasfield(typeof(code), :tensorindex)
        return [code.tensorindex]
    else
        # Pre-allocate with size hint based on number of arguments
        vs = Int[]
        sizehint!(vs, length(code.args) * 2)  # Heuristic: assume ~2 indices per arg
        
        for arg in code.args
            append!(vs, _collect_tensorindices(arg))
        end
        return vs
    end
end

function ein2contraction_tree(code)
    tids = sort(unique(_collect_tensorindices(code)))
    pos = Dict(tid => i for (i, tid) in enumerate(tids))
    return _ein2contraction_tree(code, pos)
end

function _ein2contraction_tree(code, pos)
    if hasfield(typeof(code), :tensorindex)
        return pos[code.tensorindex]
    else
        # Build tree iteratively from left to right
        t = _ein2contraction_tree(code.args[1], pos)
        for i in 2:length(code.args)
            t = ContractionTree(t, _ein2contraction_tree(code.args[i], pos))
        end
        return t
    end
end
