# transform optimized eincode to elimination order
function eincode2order(code::NestedEinsum{L}) where {L}
    elimination_order = Vector{L}()
    OMEinsum.isleaf(code) && return elimination_order
    for node in PostOrderDFS(code)
        (node isa LeafString) && continue
        for id in setdiff(vcat(getixsv(node.eins)...), getiyv(node.eins))
            push!(elimination_order, id)
        end
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
    nodes = Vector{Union{ContractionTree, Int}}(leaves)
    while length(nodes) > 1
        next = Union{ContractionTree, Int}[]
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
    eo2ct(grouped_eo, incidence_list, weights) -> ContractionTree

Construct a contraction tree from a grouped elimination order using the
current OMEinsumContractionOrders data structures.

Arguments
- `grouped_eo`: vector of vertex groups (each group is a vector of graph vertex ids)
- `incidence_list`: incidence structure mapping tensor indices to labels and
  providing `e2v::Dict{Int, Vector{Int}}` from vertex id to tensor indices
- `weights`: kept for backward-compatibility; not used by the builder

Returns
- `ContractionTree` combining all tensors associated with each group, then
  combining groups into a single binary tree

Example
```julia
rcode = rawcode(IndependentSet(g))
ixs = getixsv(rcode)
incidence_list = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]))
grouped_eo = [[i] for i in eo]
ct = eo2ct(grouped_eo, incidence_list, [1.0 for _ in 1:length(eo)])
code = decorate(parse_eincode(incidence_list, ct, vertices=collect(1:length(ixs))))
```
"""
function eo2ct(grouped_eo::Vector{<:AbstractVector{Int}}, incidence_list::IncidenceList, weights)
    trees = Union{ContractionTree, Int}[]
    for grp in grouped_eo
        leaves = Int[]
        for v in grp
            push!(leaves, v)
        end
        push!(trees, build_balanced_tree(leaves))
    end
    return reduce((x,y) -> ContractionTree(x, y), trees)
end



function decompose(code::NestedEinsum{L}) where {L}
    g, id_dict = eincode2graph(code)
    labels = collect(keys(id_dict))[sortperm(collect(values(id_dict)))]
    return decomposition_tree(g, eincode2order(code), labels = labels)
end

function max_bag(tree::DecompositionTreeNode)
    max_bag = tree.bag
    max_size = length(max_bag)
    for node in PostOrderDFS(tree)
        if length(node.bag) > max_size
            max_bag = node.bag
            max_size = length(node.bag)
        end
    end
    return max_bag
end

# this function maps an elimination order on a old graph to a new graph with some vertices removed or reordered
function update_order(eo_old::Vector{Int}, vmap::Vector{Int})
    ivmap = inverse_vmap_dict(vmap)
    eo_new = Vector{Int}()
    for v in eo_old
        haskey(ivmap, v) && push!(eo_new, ivmap[v])
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
        vs = Int[]
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
        t = _ein2contraction_tree(code.args[1], pos)
        for i in 2:length(code.args)
            t = ContractionTree(t, _ein2contraction_tree(code.args[i], pos))
        end
        return t
    end
end
