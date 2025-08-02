%%%-------------------------------------------------------------------
%%% @doc DUA Graph Core Module
%%%
%%% This module handles the core graph operations including topology
%%% management, cycle detection, and graph traversal algorithms.
%%%-------------------------------------------------------------------

-module(dua_graph).

-include("dua.hrl").

-export([
    %% Graph operations
    new/0,
    new/1,
    add_node/3,
    add_edge/4,
    remove_node/2,
    remove_edge/3,

    %% Graph queries
    get_node/2,
    get_parents/2,
    get_children/2,
    get_ancestors/2,
    get_descendants/2,

    %% Topology operations
    topological_sort/1,
    has_cycle/1,
    find_path/3,
    get_markov_blanket/2,

    %% Graph analysis
    is_connected/3,
    get_connected_components/1,
    get_graph_metrics/1
]).

%%%===================================================================
%%% Graph Operations
%%%===================================================================

%% @doc Create a new empty graph
-spec new() -> #dua_graph{}.
new() ->
    new(#{}).

%% @doc Create a new graph with metadata
-spec new(map()) -> #dua_graph{}.
new(Metadata) ->
    #dua_graph{
        nodes = #{},
        edges = [],
        topology = directed_acyclic_graph,
        metadata = Metadata
    }.

%% @doc Add a node to the graph
-spec add_node(#dua_graph{}, node_id(), #dua_node{}) ->
    {ok, #dua_graph{}} | dua_error().
add_node(#dua_graph{nodes = Nodes} = Graph, NodeId, Node) ->
    case maps:is_key(NodeId, Nodes) of
        true ->
            {error, {node_exists, NodeId}};
        false ->
            UpdatedNodes = maps:put(NodeId, Node, Nodes),
            {ok, Graph#dua_graph{nodes = UpdatedNodes}}
    end.

%% @doc Add an edge between two nodes
-spec add_edge(#dua_graph{}, node_id(), node_id(), probability()) ->
    {ok, #dua_graph{}} | dua_error().
add_edge(#dua_graph{nodes = Nodes, edges = Edges} = Graph, From, To, Weight) ->
    case {maps:is_key(From, Nodes), maps:is_key(To, Nodes)} of
        {false, _} ->
            {error, {node_not_found, From}};
        {_, false} ->
            {error, {node_not_found, To}};
        {true, true} ->
            case would_create_cycle(Graph, From, To) of
                true ->
                    {error, {cycle_detected, {From, To}}};
                false ->
                    Edge = #dua_edge{from = From, to = To, weight = Weight},
                    UpdatedGraph = update_node_relationships(Graph, From, To),
                    {ok, UpdatedGraph#dua_graph{edges = [Edge | Edges]}}
            end
    end.

%% @doc Remove a node from the graph
-spec remove_node(#dua_graph{}, node_id()) -> {ok, #dua_graph{}} | dua_error().
remove_node(#dua_graph{nodes = Nodes, edges = Edges} = Graph, NodeId) ->
    case maps:is_key(NodeId, Nodes) of
        false ->
            {error, {node_not_found, NodeId}};
        true ->
            %% Remove all edges involving this node
            FilteredEdges = lists:filter(fun(#dua_edge{from = F, to = T}) ->
                F =/= NodeId andalso T =/= NodeId
            end, Edges),

            %% Update parent-child relationships
            UpdatedGraph = remove_node_relationships(Graph, NodeId),

            %% Remove the node
            UpdatedNodes = maps:remove(NodeId, Nodes),

            {ok, UpdatedGraph#dua_graph{
                nodes = UpdatedNodes,
                edges = FilteredEdges
            }}
    end.

%% @doc Remove an edge from the graph
-spec remove_edge(#dua_graph{}, node_id(), node_id()) ->
    {ok, #dua_graph{}} | dua_error().
remove_edge(#dua_graph{edges = Edges} = Graph, From, To) ->
    case lists:keyfind({From, To}, 1,
                      [{E#dua_edge.from, E#dua_edge.to} || E <- Edges]) of
        false ->
            {error, {edge_not_found, {From, To}}};
        _ ->
            FilteredEdges = lists:filter(fun(#dua_edge{from = F, to = T}) ->
                not (F =:= From andalso T =:= To)

            end, Edges),

            UpdatedGraph = remove_edge_relationship(Graph, From, To),
            {ok, UpdatedGraph#dua_graph{edges = FilteredEdges}}
    end.

%%%===================================================================
%%% Graph Queries
%%%===================================================================

%% @doc Get a node by ID
-spec get_node(#dua_graph{}, node_id()) -> {ok, #dua_node{}} | dua_error().
get_node(#dua_graph{nodes = Nodes}, NodeId) ->
    case maps:get(NodeId, Nodes, undefined) of
        undefined -> {error, {node_not_found, NodeId}};
        Node -> {ok, Node}
    end.

%% @doc Get all parent nodes of a given node
-spec get_parents(#dua_graph{}, node_id()) -> {ok, [node_id()]} | dua_error().
get_parents(Graph, NodeId) ->
    case get_node(Graph, NodeId) of
        {ok, #dua_node{parents = Parents}} -> {ok, Parents};
        Error -> Error
    end.

%% @doc Get all children nodes of a given node
-spec get_children(#dua_graph{}, node_id()) -> {ok, [node_id()]} | dua_error().
get_children(Graph, NodeId) ->
    case get_node(Graph, NodeId) of
        {ok, #dua_node{children = Children}} -> {ok, Children};
        Error -> Error
    end.

%% @doc Get all ancestor nodes (recursive parents)
-spec get_ancestors(#dua_graph{}, node_id()) -> {ok, [node_id()]} | dua_error().
get_ancestors(Graph, NodeId) ->
    case get_parents(Graph, NodeId) of
        {ok, Parents} ->
            Ancestors = collect_ancestors(Graph, Parents, sets:new()),
            {ok, sets:to_list(Ancestors)};
        Error -> Error
    end.

%% @doc Get all descendant nodes (recursive children)
-spec get_descendants(#dua_graph{}, node_id()) -> {ok, [node_id()]} | dua_error().
get_descendants(Graph, NodeId) ->
    case get_children(Graph, NodeId) of
        {ok, Children} ->
            Descendants = collect_descendants(Graph, Children, sets:new()),
            {ok, sets:to_list(Descendants)};
        Error -> Error
    end.

%%%===================================================================
%%% Topology Operations
%%%===================================================================

%% @doc Perform topological sort of the graph
-spec topological_sort(#dua_graph{}) -> {ok, [node_id()]} | dua_error().
topological_sort(#dua_graph{nodes = Nodes} = Graph) ->
    case has_cycle(Graph) of
        true ->
            {error, graph_has_cycles};
        false ->
            NodeIds = maps:keys(Nodes),
            {ok, kahn_sort(Graph, NodeIds)}
    end.

%% @doc Check if the graph has cycles
-spec has_cycle(#dua_graph{}) -> boolean().
has_cycle(#dua_graph{nodes = Nodes} = Graph) ->
    NodeIds = maps:keys(Nodes),
    detect_cycle_dfs(Graph, NodeIds, sets:new(), sets:new()).

%% @doc Find a path between two nodes
-spec find_path(#dua_graph{}, node_id(), node_id()) ->
    {ok, [node_id()]} | {error, no_path}.
find_path(Graph, From, To) ->
    bfs_path(Graph, From, To, queue:in(From, queue:new()), sets:new(), #{}).

%% @doc Get the Markov blanket of a node
-spec get_markov_blanket(#dua_graph{}, node_id()) ->
    {ok, [node_id()]} | dua_error().
get_markov_blanket(Graph, NodeId) ->
    case get_node(Graph, NodeId) of
        {error, _} = Error -> Error;
        {ok, #dua_node{parents = Parents, children = Children}} ->
            %% Markov blanket = parents + children + parents of children
            ChildrenParents = lists:foldl(fun(Child, Acc) ->
                case get_parents(Graph, Child) of
                    {ok, CP} -> CP ++ Acc;
                    _ -> Acc
                end
            end, [], Children),

            Blanket = lists:usort(Parents ++ Children ++ ChildrenParents) -- [NodeId],
            {ok, Blanket}
    end.

%%%===================================================================
%%% Graph Analysis
%%%===================================================================

%% @doc Check if two nodes are connected (undirected)
-spec is_connected(#dua_graph{}, node_id(), node_id()) -> boolean().
is_connected(Graph, NodeA, NodeB) ->
    case find_path(Graph, NodeA, NodeB) of
        {ok, _} -> true;
        _ ->
            case find_path(Graph, NodeB, NodeA) of
                {ok, _} -> true;
                _ -> false
            end
    end.

%% @doc Get connected components of the graph
-spec get_connected_components(#dua_graph{}) -> [[node_id()]].
get_connected_components(#dua_graph{nodes = Nodes} = Graph) ->
    NodeIds = maps:keys(Nodes),
    find_components(Graph, NodeIds, []).

%% @doc Get various metrics about the graph
-spec get_graph_metrics(#dua_graph{}) -> map().
get_graph_metrics(#dua_graph{nodes = Nodes, edges = Edges}) ->
    NumNodes = maps:size(Nodes),
    NumEdges = length(Edges),

    %% Calculate node degrees
    {InDegrees, OutDegrees} = calculate_degrees(Edges, Nodes),

    #{
        num_nodes => NumNodes,
        num_edges => NumEdges,
        density => if NumNodes > 1 -> NumEdges / (NumNodes * (NumNodes - 1));
                     true -> 0.0
                   end,
        avg_in_degree => maps:fold(fun(_, D, Sum) -> Sum + D end, 0, InDegrees) / NumNodes,
        avg_out_degree => maps:fold(fun(_, D, Sum) -> Sum + D end, 0, OutDegrees) / NumNodes,
        max_in_degree => lists:max([0 | maps:values(InDegrees)]),
        max_out_degree => lists:max([0 | maps:values(OutDegrees)])
    }.

%%%===================================================================
%%% Internal Functions
%%%===================================================================

%% @private
would_create_cycle(Graph, From, To) ->
    case find_path(Graph, To, From) of
        {ok, _} -> true;
        _ -> false
    end.

%% @private
update_node_relationships(#dua_graph{nodes = Nodes} = Graph, From, To) ->
    FromNode = maps:get(From, Nodes),
    ToNode = maps:get(To, Nodes),

    UpdatedFromNode = FromNode#dua_node{
        children = lists:usort([To | FromNode#dua_node.children])
    },
    UpdatedToNode = ToNode#dua_node{
        parents = lists:usort([From | ToNode#dua_node.parents])
    },

    UpdatedNodes = maps:put(From, UpdatedFromNode,
                           maps:put(To, UpdatedToNode, Nodes)),

    Graph#dua_graph{nodes = UpdatedNodes}.

%% @private
remove_node_relationships(#dua_graph{nodes = Nodes} = Graph, NodeId) ->
    {ok, #dua_node{parents = Parents, children = Children}} = get_node(Graph, NodeId),

    %% Remove NodeId from children of all parents
    UpdatedNodes = lists:foldl(fun(ParentId, NodesAcc) ->
        Parent = maps:get(ParentId, NodesAcc),
        UpdatedParent = Parent#dua_node{
            children = lists:delete(NodeId, Parent#dua_node.children)
        },
        maps:put(ParentId, UpdatedParent, NodesAcc)
    end, Nodes, Parents),

    %% Remove NodeId from parents of all children
    FinalNodes = lists:foldl(fun(ChildId, NodesAcc) ->
        Child = maps:get(ChildId, NodesAcc),
        UpdatedChild = Child#dua_node{
            parents = lists:delete(NodeId, Child#dua_node.parents)
        },
        maps:put(ChildId, UpdatedChild, NodesAcc)
    end, UpdatedNodes, Children),

    Graph#dua_graph{nodes = FinalNodes}.

%% @private
remove_edge_relationship(#dua_graph{nodes = Nodes} = Graph, From, To) ->
    FromNode = maps:get(From, Nodes),
    ToNode = maps:get(To, Nodes),

    UpdatedFromNode = FromNode#dua_node{
        children = lists:delete(To, FromNode#dua_node.children)
    },
    UpdatedToNode = ToNode#dua_node{
        parents = lists:delete(From, ToNode#dua_node.parents)
    },

    UpdatedNodes = maps:put(From, UpdatedFromNode,
                           maps:put(To, UpdatedToNode, Nodes)),

    Graph#dua_graph{nodes = UpdatedNodes}.

%% @private
collect_ancestors(Graph, ParentIds, Visited) ->
    lists:foldl(fun(ParentId, VisitedAcc) ->
        case sets:is_element(ParentId, VisitedAcc) of
            true -> VisitedAcc;
            false ->
                NewVisited = sets:add_element(ParentId, VisitedAcc),
                case get_parents(Graph, ParentId) of
                    {ok, GrandParents} ->
                        collect_ancestors(Graph, GrandParents, NewVisited);
                    _ -> NewVisited
                end
        end
    end, Visited, ParentIds).

%% @private
collect_descendants(Graph, ChildIds, Visited) ->
    lists:foldl(fun(ChildId, VisitedAcc) ->
        case sets:is_element(ChildId, VisitedAcc) of
            true -> VisitedAcc;
            false ->
                NewVisited = sets:add_element(ChildId, VisitedAcc),
                case get_children(Graph, ChildId) of
                    {ok, GrandChildren} ->
                        collect_descendants(Graph, GrandChildren, NewVisited);
                    _ -> NewVisited
                end
        end
    end, Visited, ChildIds).

%% @private
kahn_sort(Graph, NodeIds) ->
    %% Calculate in-degrees
    InDegrees = lists:foldl(fun(NodeId, Acc) ->
        {ok, Parents} = get_parents(Graph, NodeId),
        maps:put(NodeId, length(Parents), Acc)
    end, #{}, NodeIds),

    %% Find nodes with no incoming edges
    Queue = queue:from_list([N || N <- NodeIds, maps:get(N, InDegrees) =:= 0]),

    kahn_sort_loop(Graph, Queue, InDegrees, []).

%% @private
kahn_sort_loop(_Graph, Queue, _InDegrees, Result) ->
    case queue:is_empty(Queue) of
        true -> lists:reverse(Result);
        false ->
            {{value, Node}, RestQueue} = queue:out(Queue),
            {ok, Children} = get_children(_Graph, Node),

            {NewQueue, NewInDegrees} = lists:foldl(fun(Child, {QAcc, InAcc}) ->
                NewInDegree = maps:get(Child, InAcc) - 1,
                UpdatedInAcc = maps:put(Child, NewInDegree, InAcc),
                if NewInDegree =:= 0 ->
                    {queue:in(Child, QAcc), UpdatedInAcc};
                true ->
                    {QAcc, UpdatedInAcc}
                end
            end, {RestQueue, _InDegrees}, Children),

            kahn_sort_loop(_Graph, NewQueue, NewInDegrees, [Node | Result])
    end.

%% @private
detect_cycle_dfs(_Graph, [], _Visited, _RecStack) ->
    false;
detect_cycle_dfs(Graph, [Node | Rest], Visited, RecStack) ->
    case sets:is_element(Node, Visited) of
        true ->
            detect_cycle_dfs(Graph, Rest, Visited, RecStack);
        false ->
            case dfs_cycle_check(Graph, Node, Visited, RecStack) of
                true -> true;
                false ->
                    detect_cycle_dfs(Graph, Rest, sets:add_element(Node, Visited), RecStack)
            end
    end.

%% @private
dfs_cycle_check(Graph, Node, Visited, RecStack) ->
    NewVisited = sets:add_element(Node, Visited),
    NewRecStack = sets:add_element(Node, RecStack),

    {ok, Children} = get_children(Graph, Node),
    check_children_for_cycle(Graph, Children, NewVisited, NewRecStack).

%% @private
check_children_for_cycle(_Graph, [], _Visited, _RecStack) ->
    false;
check_children_for_cycle(Graph, [Child | Rest], Visited, RecStack) ->
    case sets:is_element(Child, RecStack) of
        true -> true;
        false ->
            case sets:is_element(Child, Visited) of
                true ->
                    check_children_for_cycle(Graph, Rest, Visited, RecStack);
                false ->
                    case dfs_cycle_check(Graph, Child, Visited, RecStack) of
                        true -> true;
                        false ->
                            check_children_for_cycle(Graph, Rest, Visited, RecStack)
                    end
            end
    end.

%% @private
bfs_path(_Graph, From, To, _Queue, _Visited, Parents) when From =:= To ->
    Path = reconstruct_path(To, Parents, [To]),
    {ok, Path};
bfs_path(Graph, From, To, Queue, Visited, Parents) ->
    case queue:out(Queue) of
        {empty, _} ->
            {error, no_path};
        {{value, Current}, RestQueue} ->
            case sets:is_element(Current, Visited) of
                true ->
                    bfs_path(Graph, From, To, RestQueue, Visited, Parents);
                false ->
                    NewVisited = sets:add_element(Current, Visited),
                    {ok, Children} = get_children(Graph, Current),

                    {NewQueue, NewParents} = lists:foldl(fun(Child, {QAcc, PAcc}) ->
                        case sets:is_element(Child, NewVisited) of
                            true -> {QAcc, PAcc};
                            false ->
                                {queue:in(Child, QAcc), maps:put(Child, Current, PAcc)}
                        end
                    end, {RestQueue, Parents}, Children),

                    bfs_path(Graph, From, To, NewQueue, NewVisited, NewParents)
            end
    end.

%% @private
reconstruct_path(Node, Parents, Path) ->
    case maps:get(Node, Parents, undefined) of
        undefined -> Path;
        Parent -> reconstruct_path(Parent, Parents, [Parent | Path])
    end.

%% @private
find_components(_Graph, [], Components) ->
    Components;
find_components(Graph, [Node | Rest], Components) ->
    case lists:any(fun(Component) -> lists:member(Node, Component) end, Components) of
        true ->
            find_components(Graph, Rest, Components);
        false ->
            Component = find_component(Graph, Node, sets:new()),
            NewComponents = [sets:to_list(Component) | Components],
            find_components(Graph, Rest, NewComponents)
    end.

%% @private
find_component(Graph, Node, Visited) ->
    case sets:is_element(Node, Visited) of
        true -> Visited;
        false ->
            NewVisited = sets:add_element(Node, Visited),
            {ok, Children} = get_children(Graph, Node),
            {ok, Parents} = get_parents(Graph, Node),

            Neighbors = Children ++ Parents,
            lists:foldl(fun(Neighbor, VisitedAcc) ->
                find_component(Graph, Neighbor, VisitedAcc)
            end, NewVisited, Neighbors)
    end.

%% @private
calculate_degrees(Edges, Nodes) ->
    NodeIds = maps:keys(Nodes),
    InitialDegrees = maps:from_list([{N, 0} || N <- NodeIds]),

    lists:foldl(fun(#dua_edge{from = From, to = To}, {InAcc, OutAcc}) ->
        NewInAcc = maps:put(To, maps:get(To, InAcc, 0) + 1, InAcc),
        NewOutAcc = maps:put(From, maps:get(From, OutAcc, 0) + 1, OutAcc),
        {NewInAcc, NewOutAcc}
    end, {InitialDegrees, InitialDegrees}, Edges).
