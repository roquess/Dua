%%%-------------------------------------------------------------------
%%% @doc DUA Belief Update Engine
%%
%%% This module handles dynamic belief updates, evidence propagation,
%%% and incremental inference when new information becomes available.
%%%-------------------------------------------------------------------
-module(dua_update).
-include("dua.hrl").

%% API
-export([
    %% Main update functions
    propagate_evidence/2,
    propagate_evidence/3,
    update_belief/3,
    batch_update/2,
    %% Incremental inference
    incremental_forward/3,
    incremental_backward/3,
    lazy_propagation/3,
    %% Evidence handling
    add_evidence/3,
    remove_evidence/2,
    update_evidence/3,
    conflicting_evidence/2,
    %% Belief revision
    revise_beliefs/3,
    temporal_update/4,
    weighted_update/4,
    %% Utilities
    compute_belief_change/2,
    measure_update_impact/3,
    validate_update/2
]).

%%%===================================================================
%%% Main Update Functions
%%%===================================================================

%% @doc Propagate evidence using default algorithm ('pearl_propagation')
%% @param Graph The DUA graph to update.
%% @param Evidence Map of node_id() => node_state() to propagate.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec propagate_evidence(#dua_graph{}, #{node_id() => node_state()}) -> {ok, #dua_graph{}} | dua_error().
propagate_evidence(Graph, Evidence) ->
    propagate_evidence(Graph, Evidence, #{algorithm => pearl_propagation}).

%% @doc Propagate evidence with options (specifying algorithm, etc)
%% @param Options Map of options, e.g. #{algorithm => Algorithm, max_iterations => MaxIterations}.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec propagate_evidence(#dua_graph{}, #{node_id() => node_state()}, map()) -> {ok, #dua_graph{}} | dua_error().
propagate_evidence(Graph, Evidence, Options) ->
    Algorithm = maps:get(algorithm, Options, pearl_propagation),
    case validate_evidence(Graph, Evidence) of
        ok ->
            execute_propagation(Graph, Evidence, Algorithm, Options);
        Error ->
            Error
    end.

%% @doc Update a node's belief and propagate the change locally
%% @param NodeId The ID of the node to update.
%% @param NewBelief The new belief to assign.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec update_belief(#dua_graph{}, node_id(), belief()) -> {ok, #dua_graph{}} | dua_error().
update_belief(#dua_graph{nodes = Nodes} = Graph, NodeId, NewBelief) ->
    case maps:get(NodeId, Nodes, undefined) of
        undefined ->
            {error, {node_not_found, NodeId}};
        Node ->
            case dua_belief:is_valid_belief(NewBelief) of
                true ->
                    UpdatedNode = Node#dua_node{current_belief = NewBelief},
                    UpdatedNodes = maps:put(NodeId, UpdatedNode, Nodes),
                    UpdatedGraph = Graph#dua_graph{nodes = UpdatedNodes},
                    propagate_local_update(UpdatedGraph, NodeId);
                false ->
                    {error, {invalid_belief, NewBelief}}
            end
    end.

%% @doc Perform a batch update of multiple node beliefs
%% @param BeliefUpdates Map of node_id() => belief() to update.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec batch_update(#dua_graph{}, #{node_id() => belief()}) -> {ok, #dua_graph{}} | dua_error().
batch_update(Graph, BeliefUpdates) ->
    try
        %% Validate all beliefs before batch update
        maps:fold(
            fun(_NodeId, Belief, ok) ->
                case dua_belief:is_valid_belief(Belief) of
                    true -> ok;
                    false -> throw({invalid_belief, Belief})
                end;
            (_, _, Error) -> throw(Error)
            end,
            ok,
            BeliefUpdates
        ),
        %% Sequentially apply updates
        UpdatedGraph = maps:fold(
            fun(NodeId, Belief, AccGraph) ->
                case update_belief_direct(AccGraph, NodeId, Belief) of
                    {ok, NewGraph} -> NewGraph;
                    {error, Reason} -> throw(Reason)
                end
            end,
            Graph,
            BeliefUpdates
        ),
        %% After all updates, do a global propagation step
        global_propagation(UpdatedGraph)
    catch
        throw:Error -> {error, Error}
    end.

%%%===================================================================
%%% Incremental Inference
%%%===================================================================

%% @doc Incrementally update beliefs when evidence changes (forward)
%% @param OldEvidence Previous evidence map.
%% @param NewEvidence Updated evidence map.
%% @return {ok, query_result()} if successful, otherwise dua_error().
-spec incremental_forward(#dua_graph{}, #{node_id() => node_state()}, #{node_id() => node_state()}) ->
    {ok, query_result()} | dua_error().
incremental_forward(Graph, OldEvidence, NewEvidence) ->
    try
        {Added, Removed, Modified} = compute_evidence_diff(OldEvidence, NewEvidence),
        {ok, UpdatedGraph} = apply_incremental_changes(Graph, Added, Removed, Modified),
        case identify_affected_nodes(UpdatedGraph, NewEvidence) of
            {ok, AffectedNodes} ->
                targeted_forward_inference(UpdatedGraph, NewEvidence, AffectedNodes);
            Error -> Error
        end
    catch
        error:Reason ->
            {error, {incremental_forward_failed, Reason}}
    end.

%% @doc Incrementally update beliefs when evidence changes (backward)
%% @param OldEvidence Previous evidence map.
%% @param NewEvidence Updated evidence map.
%% @return {ok, query_result()} if successful, otherwise dua_error().
-spec incremental_backward(#dua_graph{}, #{node_id() => node_state()}, #{node_id() => node_state()}) ->
    {ok, query_result()} | dua_error().
incremental_backward(Graph, OldEvidence, NewEvidence) ->
    try
        {Added, Removed, Modified} = compute_evidence_diff(OldEvidence, NewEvidence),
        {ok, UpdatedGraph} = apply_incremental_changes(Graph, Added, Removed, Modified),
        {ok, NewCauses} = identify_new_causes(UpdatedGraph, Added),
        targeted_backward_inference(UpdatedGraph, NewEvidence, NewCauses)
    catch
        error:Reason ->
            {error, {incremental_backward_failed, Reason}}
    end.

%% @doc Lazy propagation: beliefs are updated for query nodes on demand
%% @param QueryNodes List of nodes whose beliefs are to be updated.
%% @return {ok, #{node_id() => belief()}} if successful, otherwise dua_error().
-spec lazy_propagation(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) ->
    {ok, #{node_id() => belief()}} | dua_error().
lazy_propagation(Graph, Evidence, QueryNodes) ->
    try
        {ok, PropagationQueue} = build_lazy_queue(Graph, Evidence, QueryNodes),
        {ok, UpdatedGraph} = lazy_propagate_queue(Graph, Evidence, PropagationQueue),
        QueryBeliefs = maps:fold(fun(NodeId, Node, Acc) ->
            case lists:member(NodeId, QueryNodes) of
                true -> maps:put(NodeId, Node#dua_node.current_belief, Acc);
                false -> Acc
            end
        end, #{}, UpdatedGraph#dua_graph.nodes),
        {ok, QueryBeliefs}
    catch
        error:Reason -> {error, {lazy_propagation_failed, Reason}}
    end.

%%%===================================================================
%%% Evidence Handling
%%%===================================================================

%% @doc Add new evidence for a node (assign state and propagate)
%% @param NodeId The ID of the node.
%% @param EvidenceState The state to set as evidence.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec add_evidence(#dua_graph{}, node_id(), node_state()) -> {ok, #dua_graph{}} | dua_error().
add_evidence(Graph, NodeId, EvidenceState) ->
    case dua_graph:get_node(Graph, NodeId) of
        {ok, #dua_node{states = States}} ->
            case lists:member(EvidenceState, States) of
                true ->
                    Evidence = #{NodeId => EvidenceState},
                    propagate_evidence(Graph, Evidence);
                false ->
                    {error, {invalid_evidence_state, NodeId, EvidenceState}}
            end;
        Error -> Error
    end.

%% @doc Remove evidence from a node (reset to uniform and propagate removal)
%% @param NodeId The ID of the node.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec remove_evidence(#dua_graph{}, node_id()) -> {ok, #dua_graph{}} | dua_error().
remove_evidence(#dua_graph{nodes = Nodes} = Graph, NodeId) ->
    case maps:get(NodeId, Nodes, undefined) of
        undefined ->
            {error, {node_not_found, NodeId}};
        #dua_node{states = States} = Node ->
            UniformBelief = dua_belief:uniform_belief(States),
            UpdatedNode = Node#dua_node{current_belief = UniformBelief},
            UpdatedNodes = maps:put(NodeId, UpdatedNode, Nodes),
            UpdatedGraph = Graph#dua_graph{nodes = UpdatedNodes},
            propagate_evidence_removal(UpdatedGraph, NodeId)
    end.

%% @doc Update evidence on a node (remove old, add new)
%% @param NodeId The ID of the node.
%% @param NewEvidenceState The new state to set as evidence.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec update_evidence(#dua_graph{}, node_id(), node_state()) -> {ok, #dua_graph{}} | dua_error().
update_evidence(Graph, NodeId, NewEvidenceState) ->
    case remove_evidence(Graph, NodeId) of
        {ok, IntermediateGraph} ->
            add_evidence(IntermediateGraph, NodeId, NewEvidenceState);
        Error -> Error
    end.

%% @doc Detect conflicting evidence in the graph
%% @param Evidence The evidence map to check.
%% @return {ok, [#{conflict => term(), nodes => [node_id()]}]} if successful, otherwise dua_error().
-spec conflicting_evidence(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, [#{conflict => term(), nodes => [node_id()]}]} | dua_error().
conflicting_evidence(Graph, Evidence) ->
    try
        DirectConflicts = detect_direct_conflicts(Graph, Evidence),
        ProbConflicts = detect_probabilistic_conflicts(Graph, Evidence),
        {ok, DirectConflicts ++ ProbConflicts}
    catch
        error:Reason -> {error, {conflict_detection_failed, Reason}}
    end.

%%%===================================================================
%%% Belief Revision
%%%===================================================================

%% @doc Revise beliefs according to a revision strategy and confidences
%% @param Options Map of options, e.g. #{strategy => Strategy, weights => Weights}.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec revise_beliefs(#dua_graph{}, #{node_id() => node_state()}, map()) -> {ok, #dua_graph{}} | dua_error().
revise_beliefs(Graph, NewEvidence, Options) ->
    RevisionStrategy = maps:get(strategy, Options, weighted_average),
    ConfidenceWeights = maps:get(weights, Options, #{}),
    case RevisionStrategy of
        weighted_average ->
            weighted_belief_revision(Graph, NewEvidence, ConfidenceWeights);
        jeffrey_rule ->
            jeffrey_belief_revision(Graph, NewEvidence, ConfidenceWeights);
        minimal_change ->
            minimal_change_revision(Graph, NewEvidence, ConfidenceWeights);
        _ ->
            {error, {unknown_revision_strategy, RevisionStrategy}}
    end.

%% @doc Update beliefs with temporal decay (older beliefs fade out)
%% @param DecayRate The rate at which beliefs decay (0.0 to 1.0).
%% @param TimeSteps The number of time steps to apply decay.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec temporal_update(#dua_graph{}, #{node_id() => node_state()}, float(), pos_integer()) ->
    {ok, #dua_graph{}} | dua_error().
temporal_update(Graph, Evidence, DecayRate, TimeSteps) when DecayRate >= 0.0, DecayRate =< 1.0 ->
    try
        %% Apply decay
        {ok, DecayedGraph} = apply_temporal_decay(Graph, DecayRate, TimeSteps),
        propagate_evidence(DecayedGraph, Evidence)
    catch
        error:Reason -> {error, {temporal_update_failed, Reason}}
    end;
temporal_update(_Graph, _Evidence, DecayRate, _TimeSteps) ->
    {error, {invalid_decay_rate, DecayRate}}.

%% @doc Combine evidence from multiple sources with weights
%% @param EvidenceList List of evidence maps.
%% @param Weights List of weights corresponding to each evidence map.
%% @param Options Map of options.
%% @return {ok, #dua_graph{}} if successful, otherwise dua_error().
-spec weighted_update(#dua_graph{}, [#{node_id() => node_state()}], [float()], map()) ->
    {ok, #dua_graph{}} | dua_error().
weighted_update(Graph, EvidenceList, Weights, Options) when length(EvidenceList) =:= length(Weights) ->
    try
        case lists:all(fun(W) -> W >= 0.0 end, Weights) of
            false -> throw({invalid_weights, Weights});
            true -> ok
        end,
        TotalWeight = lists:sum(Weights),
        NormalizedWeights = case TotalWeight > ?DEFAULT_PRECISION of
            true -> [W / TotalWeight || W <- Weights];
            false -> uniform_weights(length(Weights))
        end,
        {ok, CombinedEvidence} = combine_weighted_evidence(EvidenceList, NormalizedWeights),
        propagate_evidence(Graph, CombinedEvidence, Options)
    catch
        throw:Error -> {error, Error};
        error:Reason -> {error, {weighted_update_failed, Reason}}
    end;
weighted_update(_Graph, EvidenceList, Weights, _Options) ->
    {error, {length_mismatch, length(EvidenceList), length(Weights)}}.

%%%===================================================================
%%% Utilities
%%%===================================================================

%% @doc Compute average belief change per node between two graphs
%% @param OldGraph The graph before the update.
%% @param NewGraph The graph after the update.
%% @return #{node_id() => float()} map of belief changes.
-spec compute_belief_change(#dua_graph{}, #dua_graph{}) -> #{node_id() => float()}.
compute_belief_change(#dua_graph{nodes = OldNodes}, #dua_graph{nodes = NewNodes}) ->
    CommonNodes = sets:to_list(sets:intersection(sets:from_list(maps:keys(OldNodes)), sets:from_list(maps:keys(NewNodes)))),
    maps:from_list([{NodeId, compute_node_belief_change(maps:get(NodeId, OldNodes), maps:get(NodeId, NewNodes))} || NodeId <- CommonNodes]).

%% @doc Measure the impact (sum, max, avg) of an update
%% @param OldGraph The graph before the update.
%% @param NewGraph The graph after the update.
%% @param Evidence The evidence applied.
%% @return #{total_change => float(), affected_nodes => [node_id()], max_change => float(), avg_change => float(), evidence_nodes => [node_id()]}.
-spec measure_update_impact(#dua_graph{}, #dua_graph{}, #{node_id() => node_state()}) ->
    #{total_change => float(), affected_nodes => [node_id()], max_change => float(), avg_change => float(), evidence_nodes => [node_id()]}.
measure_update_impact(OldGraph, NewGraph, Evidence) ->
    Changes = compute_belief_change(OldGraph, NewGraph),
    ChangeValues = maps:values(Changes),
    AffectedNodes = [NodeId || {NodeId, Change} <- maps:to_list(Changes), Change > ?DEFAULT_PRECISION],
    TotalChange = lists:sum(ChangeValues),
    MaxChange = case ChangeValues of [] -> 0.0; _ -> lists:max(ChangeValues) end,
    AvgChange = case length(ChangeValues) of 0 -> 0.0; N -> TotalChange / N end,
    #{
        total_change => TotalChange,
        affected_nodes => AffectedNodes,
        max_change => MaxChange,
        avg_change => AvgChange,
        evidence_nodes => maps:keys(Evidence)
    }.

%% @doc Validate evidence values and target nodes
%% @param Evidence The evidence map to validate.
%% @return ok if valid, otherwise dua_error().
-spec validate_update(#dua_graph{}, #{node_id() => node_state()}) -> ok | dua_error().
validate_update(Graph, Evidence) ->
    maps:fold(fun(NodeId, State, ok) ->
        case dua_graph:get_node(Graph, NodeId) of
            {ok, #dua_node{states = States}} ->
                case lists:member(State, States) of
                    true -> ok;
                    false -> {error, {invalid_state, NodeId, State}}
                end;
            {error, _} ->
                {error, {node_not_found, NodeId}}
        end;
    (_, _, Error) -> Error
    end, ok, Evidence).

%%%===================================================================
%%% Internal Helper Functions
%%%===================================================================

%% @private Validate that evidence states are valid for nodes
-spec validate_evidence(#dua_graph{}, #{node_id() => node_state()}) -> ok | dua_error().
validate_evidence(Graph, Evidence) ->
    validate_update(Graph, Evidence).

%% @private Execute evidence propagation using the chosen algorithm
-spec execute_propagation(#dua_graph{}, #{node_id() => node_state()}, atom(), map()) ->
    {ok, #dua_graph{}} | dua_error().
execute_propagation(Graph, Evidence, Algorithm, _Options) ->
    case Algorithm of
        pearl_propagation ->
            pearl_propagation_algorithm(Graph, Evidence);
        simple_propagation ->
            simple_propagation_algorithm(Graph, Evidence);
        message_passing ->
            message_passing_algorithm(Graph, Evidence);
        _ -> {error, {unknown_algorithm, Algorithm}}
    end.

%% @private Stub: Pearl's propagation - in practice, would carry out full belief propagation
-spec pearl_propagation_algorithm(#dua_graph{}, #{node_id() => node_state()}) -> {ok, #dua_graph{}} | dua_error().
pearl_propagation_algorithm(Graph, Evidence) ->
    apply_evidence_directly(Graph, Evidence).

%% @private: Simple propagation - direct evidence application
-spec simple_propagation_algorithm(#dua_graph{}, #{node_id() => node_state()}) -> {ok, #dua_graph{}} | dua_error().
simple_propagation_algorithm(Graph, Evidence) ->
    apply_evidence_directly(Graph, Evidence).

%% @private: Message-passing (alias for Pearl/simple here)
-spec message_passing_algorithm(#dua_graph{}, #{node_id() => node_state()}) -> {ok, #dua_graph{}} | dua_error().
message_passing_algorithm(Graph, Evidence) ->
    apply_evidence_directly(Graph, Evidence).

%% @private Apply evidence to nodes: sets 'current_belief' to certain for evidence
-spec apply_evidence_directly(#dua_graph{}, #{node_id() => node_state()}) -> {ok, #dua_graph{}}.
apply_evidence_directly(#dua_graph{nodes = Nodes} = Graph, Evidence) ->
    UpdatedNodes = maps:fold(
        fun(NodeId, State, AccNodes) ->
            case maps:get(NodeId, AccNodes, undefined) of
                undefined -> AccNodes;
                Node ->
                    CertainBelief = dua_belief:certain_belief(Node#dua_node.states, State),
                    UpdatedNode = Node#dua_node{current_belief = CertainBelief},
                    maps:put(NodeId, UpdatedNode, AccNodes)
            end
        end,
        Nodes,
        Evidence
    ),
    {ok, Graph#dua_graph{nodes = UpdatedNodes}}.

%% @private Propagate local update from one node to its immediate children
-spec propagate_local_update(#dua_graph{}, node_id()) -> {ok, #dua_graph{}} | dua_error().
propagate_local_update(Graph, NodeId) ->
    case dua_graph:get_children(Graph, NodeId) of
        {ok, Children} ->
            update_children_beliefs(Graph, NodeId, Children);
        Error -> Error
    end.

%% @private: Direct update of one node's belief (no propagation)
-spec update_belief_direct(#dua_graph{}, node_id(), belief()) -> {ok, #dua_graph{}} | dua_error().
update_belief_direct(#dua_graph{nodes = Nodes} = Graph, NodeId, Belief) ->
    case maps:get(NodeId, Nodes, undefined) of
        undefined -> {error, {node_not_found, NodeId}};
        Node ->
            UpdatedNode = Node#dua_node{current_belief = Belief},
            UpdatedNodes = maps:put(NodeId, UpdatedNode, Nodes),
            {ok, Graph#dua_graph{nodes = UpdatedNodes}}
    end.

%% @private: After batch update, propagate globally (stub)
-spec global_propagation(#dua_graph{}) -> {ok, #dua_graph{}}.
global_propagation(Graph) ->
    {ok, Graph}.

%% @private: Compute distance/delta between two beliefs for a single node
-spec compute_node_belief_change(#dua_node{}, #dua_node{}) -> float().
compute_node_belief_change(#dua_node{current_belief = OldBelief}, #dua_node{current_belief = NewBelief}) ->
    dua_belief:belief_distance(OldBelief, NewBelief).

%% @private: Compute the differences (added, removed, modified) in evidence between two maps
-spec compute_evidence_diff(#{node_id() => node_state()}, #{node_id() => node_state()}) ->
    {#{node_id() => node_state()}, #{node_id() => node_state()}, #{node_id() => node_state()}}.
compute_evidence_diff(OldEvidence, NewEvidence) ->
    OldKeys = sets:from_list(maps:keys(OldEvidence)),
    NewKeys = sets:from_list(maps:keys(NewEvidence)),
    Added = maps:with(sets:to_list(sets:subtract(NewKeys, OldKeys)), NewEvidence),
    Removed = maps:with(sets:to_list(sets:subtract(OldKeys, NewKeys)), OldEvidence),
    CommonKeys = sets:to_list(sets:intersection(OldKeys, NewKeys)),
    Modified = maps:fold(
        fun(Key, NewVal, Acc) ->
            case maps:get(Key, OldEvidence) of
                NewVal -> Acc;
                _ -> maps:put(Key, NewVal, Acc)
            end
        end,
        #{}, maps:with(CommonKeys, NewEvidence)),
    {Added, Removed, Modified}.

%% @private: Apply incremental adds/removes/modifies to graph
-spec apply_incremental_changes(#dua_graph{}, #{node_id() => node_state()}, #{node_id() => node_state()}, #{node_id() => node_state()}) ->
    {ok, #dua_graph{}} | dua_error().
apply_incremental_changes(Graph, Added, Removed, Modified) ->
    {ok, G0} = maps:fold(fun(NodeId, _State, AccG) ->
        case remove_evidence(AccG, NodeId) of
            {ok, G1} -> {ok, G1};
            Error -> throw(Error)
        end
    end, {ok, Graph}, maps:keys(Removed)),
    AllAdds = maps:merge(Added, Modified),
    propagate_evidence(G0, AllAdds).

%% @private: Identify nodes whose beliefs are affected by evidence
-spec identify_affected_nodes(#dua_graph{}, #{node_id() => node_state()}) -> {ok, [node_id()]} | dua_error().
identify_affected_nodes(_Graph, Evidence) ->
    {ok, maps:keys(Evidence)}.

%% @private: Forward inference focused on a limited set of nodes
-spec targeted_forward_inference(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) ->
    {ok, #{beliefs => #{node_id() => belief()}}} | dua_error().
targeted_forward_inference(Graph, Evidence, AffectedNodes) ->
    {ok, #{beliefs := AllBeliefs}} = dua_forward:infer(Graph, Evidence),
    FilteredBeliefs = maps:with(AffectedNodes, AllBeliefs),
    {ok, #{beliefs => FilteredBeliefs}}.

%% @private: Identify newly implicated causes from evidence
-spec identify_new_causes(#dua_graph{}, #{node_id() => node_state()}) -> {ok, [node_id()]} | dua_error().
identify_new_causes(_Graph, AddedEvidence) ->
    {ok, maps:keys(AddedEvidence)}.

%% @private: Backward inference focused on a target node set
-spec targeted_backward_inference(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) ->
    {ok, #{beliefs => #{node_id() => belief()}}} | dua_error().
targeted_backward_inference(Graph, Evidence, CauseNodes) ->
    {ok, #{beliefs := AllBeliefs}} = dua_backward:infer(Graph, Evidence),
    FilteredBeliefs = maps:with(CauseNodes, AllBeliefs),
    {ok, #{beliefs => FilteredBeliefs}}.

%% @private: Build the list of nodes requiring propagation for lazy queries
-spec build_lazy_queue(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) -> {ok, [node_id()]} | dua_error().
build_lazy_queue(_Graph, _Evidence, QueryNodes) ->
    {ok, QueryNodes}.

%% @private: Perform propagation on the lazy subset
-spec lazy_propagate_queue(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) -> {ok, #dua_graph{}} | dua_error().
lazy_propagate_queue(Graph, Evidence, PropagationQueue) ->
    FilteredGraph = filter_graph_nodes(Graph, PropagationQueue),
    propagate_evidence(FilteredGraph, Evidence).

%% @private: Remove evidence, then reset children to prior
-spec propagate_evidence_removal(#dua_graph{}, node_id()) -> {ok, #dua_graph{}} | dua_error().
propagate_evidence_removal(Graph, NodeId) ->
    case dua_graph:get_children(Graph, NodeId) of
        {ok, Children} ->
            reset_children_to_prior(Graph, Children);
        Error -> Error
    end.

%% @private: Reset a set of nodes to uniform belief
-spec reset_children_to_prior(#dua_graph{}, [node_id()]) -> {ok, #dua_graph{}}.
reset_children_to_prior(Graph, Children) ->
    lists:foldl(fun(ChildId, GAcc) ->
        case dua_graph:get_node(GAcc, ChildId) of
            {ok, #dua_node{states = States} = _Node} ->
                PriorBelief = dua_belief:uniform_belief(States),
                {ok, G1} = update_belief_direct(GAcc, ChildId, PriorBelief),
                G1;
            _ -> GAcc
        end
    end, Graph, Children).

%% @private: Filter the graph to retain only specified nodes and corresponding edges
-spec filter_graph_nodes(#dua_graph{}, [node_id()]) -> #dua_graph{}.
filter_graph_nodes(#dua_graph{nodes = Nodes, edges = Edges} = Graph, KeepNodes) ->
    FilteredNodes = maps:with(KeepNodes, Nodes),
    FilteredEdges = lists:filter(fun(#dua_edge{from = From, to = To}) ->
        lists:member(From, KeepNodes) andalso lists:member(To, KeepNodes)
    end, Edges),
    Graph#dua_graph{nodes = FilteredNodes, edges = FilteredEdges}.

%% @private: Extract current node beliefs as a map
-spec extract_current_beliefs(#dua_graph{}) -> #{node_id() => belief()}.
extract_current_beliefs(#dua_graph{nodes = Nodes}) ->
    maps:map(fun(_NodeId, #dua_node{current_belief = Belief}) -> Belief end, Nodes).

%% @private: Detect impossible evidence states (stub: returns [] here)
-spec detect_direct_conflicts(#dua_graph{}, #{node_id() => node_state()}) -> [#{conflict => term(), nodes => [node_id()]}].
detect_direct_conflicts(_Graph, _Evidence) ->
    [].

%% @private: Detect very low-probability evidence (stub logic)
-spec detect_probabilistic_conflicts(#dua_graph{}, #{node_id() => node_state()}) -> [#{conflict => term(), nodes => [node_id()]}].
detect_probabilistic_conflicts(Graph, Evidence) ->
    case dua_forward:infer(Graph, Evidence) of
        {ok, #{confidence := Confidence}} when Confidence < 0.01 ->
            [#{conflict => low_probability, nodes => maps:keys(Evidence)}];
        _ -> []
    end.

%% @private: Weighted belief blending using interpolation
-spec weighted_belief_revision(#dua_graph{}, #{node_id() => node_state()}, #{node_id() => float()}) ->
    {ok, #dua_graph{}} | dua_error().
weighted_belief_revision(Graph, NewEvidence, ConfidenceWeights) ->
    DefaultWeight = maps:get(default_weight, ConfidenceWeights, 0.5),
    %% Get current beliefs
    CurrentBeliefs = extract_current_beliefs(Graph),
    case dua_forward:infer(Graph, NewEvidence) of
        {ok, #{beliefs := NewBeliefs}} ->
            CombinedBeliefs = maps:fold(
                fun(NodeId, NewBelief, Acc) ->
                    Weight = maps:get(NodeId, ConfidenceWeights, DefaultWeight),
                    OldBelief = maps:get(NodeId, CurrentBeliefs, #{}),
                    CombinedBelief = dua_belief:interpolate(OldBelief, NewBelief, Weight),
                    maps:put(NodeId, CombinedBelief, Acc)
                end, #{}, NewBeliefs),
            apply_beliefs_to_graph(Graph, CombinedBeliefs);
        Error -> Error
    end.

%% @private: Jeffrey's rule belief revision (stub, uses direct propagation)
-spec jeffrey_belief_revision(#dua_graph{}, #{node_id() => node_state()}, map()) -> {ok, #dua_graph{}} | dua_error().
jeffrey_belief_revision(Graph, NewEvidence, _Options) ->
    propagate_evidence(Graph, NewEvidence).

%% @private: Minimal change principle (stub, uses direct propagation)
-spec minimal_change_revision(#dua_graph{}, #{node_id() => node_state()}, map()) -> {ok, #dua_graph{}} | dua_error().
minimal_change_revision(Graph, NewEvidence, _Options) ->
    propagate_evidence(Graph, NewEvidence).

%% @private: Apply exponential decay to all node beliefs (toward uniform)
-spec apply_temporal_decay(#dua_graph{}, float(), pos_integer()) -> {ok, #dua_graph{}} | dua_error().
apply_temporal_decay(#dua_graph{nodes = Nodes} = Graph, DecayRate, TimeSteps) ->
    DecayFactor = math:pow(DecayRate, TimeSteps),
    UpdatedNodes = maps:map(fun(_NodeId, #dua_node{id = _Nid, current_belief = Belief, states = States} = Node) ->
        UniformBelief = dua_belief:uniform_belief(States),
        DecayedBelief = dua_belief:interpolate(UniformBelief, Belief, DecayFactor),
        Node#dua_node{current_belief = DecayedBelief}
    end, Nodes),
    {ok, Graph#dua_graph{nodes = UpdatedNodes}}.

%% @private: Return a uniform weights list of size N
-spec uniform_weights(pos_integer()) -> [float()].
uniform_weights(N) ->
    W = 1.0 / N, lists:duplicate(N, W).

%% @private: Combine multiple evidence maps using weights (very simplified: merges only strongly weighted evidence)
-spec combine_weighted_evidence([#{node_id() => node_state()}], [float()]) ->
    {ok, #{node_id() => node_state()}} | dua_error().
combine_weighted_evidence(EvidenceList, Weights) ->
    WeightedPairs = lists:zip(EvidenceList, Weights),
    CombinedEvidence = lists:foldl(fun({Evidence, Wt}, Acc) ->
        (case Wt > 0.5 of
             true -> maps:merge(Acc, Evidence);
             false -> Acc
         end)
    end, #{}, WeightedPairs),
    {ok, CombinedEvidence}.

%% @private: Update beliefs of all children after a parent was changed (stub logic)
-spec update_children_beliefs(#dua_graph{}, node_id(), [node_id()]) -> #dua_graph{}.
update_children_beliefs(Graph, ParentId, Children) ->
    lists:foldl(fun(ChildId, GAcc) ->
        case update_child_belief(GAcc, ParentId, ChildId) of
            {ok, G1} -> G1;
            _ -> GAcc
        end
    end, Graph, Children).

%% @private: Directly update a child node using parent (stub: just interpolates a bit)
-spec update_child_belief(#dua_graph{}, node_id(), node_id()) -> {ok, #dua_graph{}} | dua_error().
update_child_belief(Graph, ParentId, ChildId) ->
    case {dua_graph:get_node(Graph, ParentId), dua_graph:get_node(Graph, ChildId)} of
        {{ok, ParentNode}, {ok, ChildNode}} ->
            ParentBelief = ParentNode#dua_node.current_belief,
            ChildBelief = ChildNode#dua_node.current_belief,
            %% Simple: Blend a portion of parent into child
            NewChildBelief = dua_belief:interpolate(ChildBelief, ParentBelief, 0.3),
            update_belief_direct(Graph, ChildId, NewChildBelief);
        _ -> {ok, Graph}
    end.

%% @private: Apply a set of beliefs to the graph (replacing current_belief)
-spec apply_beliefs_to_graph(#dua_graph{}, #{node_id() => belief()}) -> {ok, #dua_graph{}} | dua_error().
apply_beliefs_to_graph(#dua_graph{nodes = Nodes} = Graph, Beliefs) ->
    UpdatedNodes = maps:fold(fun(NodeId, NewBelief, AccNodes) ->
        case maps:get(NodeId, AccNodes, undefined) of
            undefined -> AccNodes;
            Node ->
                UpdatedNode = Node#dua_node{current_belief = NewBelief},
                maps:put(NodeId, UpdatedNode, AccNodes)
        end
    end, Nodes, Beliefs),
    {ok, Graph#dua_graph{nodes = UpdatedNodes}}.

