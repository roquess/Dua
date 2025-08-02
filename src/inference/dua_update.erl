%%%-------------------------------------------------------------------
%%% @doc DUA Belief Update Engine
%%%
%%% This module handles dynamic belief updates, evidence propagation,
%%% and incremental inference when new information becomes available.
%%%-------------------------------------------------------------------

-module(dua_update).

-include("dua.hrl").

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
-spec propagate_evidence(#dua_graph{}, evidence()) -> {ok, #dua_graph{}} | dua_error().
propagate_evidence(Graph, Evidence) ->
    propagate_evidence(Graph, Evidence, #{algorithm => pearl_propagation}).

%% @doc Propagate evidence with options (specifying algorithm, etc)
-spec propagate_evidence(#dua_graph{}, evidence(), map()) -> {ok, #dua_graph{}} | dua_error().
propagate_evidence(Graph, Evidence, Options) ->
    Algorithm = maps:get(algorithm, Options, pearl_propagation),
    MaxIterations = maps:get(max_iterations, Options, ?MAX_ITERATIONS),
    case validate_evidence(Graph, Evidence) of
        ok ->
            execute_propagation(Graph, Evidence, Algorithm, Options);
        Error ->
            Error
    end.

%% @doc Update a node's belief and propagate the change locally
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
-spec incremental_forward(#dua_graph{}, evidence(), evidence()) ->
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
-spec incremental_backward(#dua_graph{}, evidence(), evidence()) ->
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
-spec lazy_propagation(#dua_graph{}, evidence(), [node_id()]) ->
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
-spec update_evidence(#dua_graph{}, node_id(), node_state()) -> {ok, #dua_graph{}} | dua_error().
update_evidence(Graph, NodeId, NewEvidenceState) ->
    case remove_evidence(Graph, NodeId) of
        {ok, IntermediateGraph} ->
            add_evidence(IntermediateGraph, NodeId, NewEvidenceState);
        Error -> Error
    end.

%% @doc Detect conflicting evidence in the graph
-spec conflicting_evidence(#dua_graph{}, evidence()) ->
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
-spec revise_beliefs(#dua_graph{}, evidence(), map()) -> {ok, #dua_graph{}} | dua_error().
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
-spec temporal_update(#dua_graph{}, evidence(), float(), pos_integer()) ->
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
-spec weighted_update(#dua_graph{}, [evidence()], [float()], map()) ->
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
-spec compute_belief_change(#dua_graph{}, #dua_graph{}) -> #{node_id() => float()}.
compute_belief_change(#dua_graph{nodes = OldNodes}, #dua_graph{nodes = NewNodes}) ->
    CommonNodes = sets:to_list(sets:intersection(sets:from_list(maps:keys(OldNodes)), sets:from_list(maps:keys(NewNodes)))),
    maps:from_list([{NodeId, compute_node_belief_change(maps:get(NodeId, OldNodes), maps:get(NodeId, NewNodes))} || NodeId <- CommonNodes]).

%% @doc Measure the impact (sum, max, avg) of an update
-spec measure_update_impact(#dua_graph{}, #dua_graph{}, evidence()) ->
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
-spec validate_update(#dua_graph{}, evidence()) -> ok | dua_error().
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
validate_evidence(Graph, Evidence) ->
    validate_update(Graph, Evidence).

%% @private Execute evidence propagation using the chosen algorithm
execute_propagation(Graph, Evidence, Algorithm, Options) ->
    case Algorithm of
        pearl_propagation ->
            pearl_propagation_algorithm(Graph, Evidence, Options);
        simple_propagation ->
            simple_propagation_algorithm(Graph, Evidence, Options);
        message_passing ->
            message_passing_algorithm(Graph, Evidence, Options);
        _ -> {error, {unknown_algorithm, Algorithm}}
    end.

%% @private Stub: Pearl's propagation - in practice, would carry out full belief propagation
pearl_propagation_algorithm(Graph, Evidence, _Options) ->
    apply_evidence_directly(Graph, Evidence).

%% @private: Simple propagation - direct evidence application
simple_propagation_algorithm(Graph, Evidence, _Options) ->
    apply_evidence_directly(Graph, Evidence).

%% @private: Message-passing (alias for Pearl/simple here)
message_passing_algorithm(Graph, Evidence, _Options) ->
    apply_evidence_directly(Graph, Evidence).

%% @private Apply evidence to nodes: sets 'current_belief' to certain for evidence
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
propagate_local_update(Graph, NodeId) ->
    case dua_graph:get_children(Graph, NodeId) of
        {ok, Children} ->
            update_children_beliefs(Graph, NodeId, Children);
        Error -> Error
    end.

%% @private: Direct update of one node's belief (no propagation)
update_belief_direct(#dua_graph{nodes = Nodes} = Graph, NodeId, Belief) ->
    case maps:get(NodeId, Nodes, undefined) of
        undefined -> {error, {node_not_found, NodeId}};
        Node ->
            UpdatedNode = Node#dua_node{current_belief = Belief},
            UpdatedNodes = maps:put(NodeId, UpdatedNode, Nodes),
            {ok, Graph#dua_graph{nodes = UpdatedNodes}}
    end.

%% @private: After batch update, propagate globally (stub)
global_propagation(Graph) ->
    {ok, Graph}.

%% @private: Compute distance/delta between two beliefs for a single node
compute_node_belief_change(#dua_node{current_belief = OldBelief}, #dua_node{current_belief = NewBelief}) ->
    dua_belief:belief_distance(OldBelief, NewBelief).

%% ===== Incomplete internal stubs for your inference/integration logic ====

%% @private: Compute the differences (added, removed, modified) in evidence between two maps
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
apply_incremental_changes(Graph, Added, Removed, Modified) ->
    {ok, G0} = maps:fold(fun(NodeId, _State, AccG) ->
        case remove_evidence(AccG, NodeId) of
            {ok, G1} -> {ok, G1};
            Error -> throw(Error)
        end
    end, {ok, Graph}, Removed),
    AllAdds = maps:merge(Added, Modified),
    propagate_evidence(G0, AllAdds).

%% @private: Identify nodes whose beliefs are affected by evidence
identify_affected_nodes(Graph, Evidence) ->
    EvidenceNodes = maps:keys(Evidence),
    %% Stub: in practice, find descendants of evidence nodes
    {ok, EvidenceNodes}.

%% @private: Forward inference focused on a limited set of nodes
targeted_forward_inference(Graph, Evidence, AffectedNodes) ->
    {ok, #{beliefs := AllBeliefs}} = dua_forward:infer(Graph, Evidence),
    FilteredBeliefs = maps:with(AffectedNodes, AllBeliefs),
    {ok, #{beliefs => FilteredBeliefs}}.

%% @private: Identify newly implicated causes from evidence
identify_new_causes(Graph, AddedEvidence) ->
    AddedNodes = maps:keys(AddedEvidence),
    {ok, AddedNodes}.

%% @private: Backward inference focused on a target node set
targeted_backward_inference(Graph, Evidence, CauseNodes) ->
    {ok, #{beliefs := AllBeliefs}} = dua_backward:infer(Graph, Evidence),
    FilteredBeliefs = maps:with(CauseNodes, AllBeliefs),
    {ok, #{beliefs => FilteredBeliefs}}.

%% @private: Build the list of nodes requiring propagation for lazy queries
build_lazy_queue(_Graph, _Evidence, QueryNodes) ->
    {ok, QueryNodes}.

%% @private: Perform propagation on the lazy subset
lazy_propagate_queue(Graph, Evidence, PropagationQueue) ->
    FilteredGraph = filter_graph_nodes(Graph, PropagationQueue),
    propagate_evidence(FilteredGraph, Evidence).

%% @private: Remove evidence, then reset children to prior
propagate_evidence_removal(Graph, NodeId) ->
    case dua_graph:get_children(Graph, NodeId) of
        {ok, Children} ->
            reset_children_to_prior(Graph, Children);
        Error -> Error
    end.

%% @private: Reset a set of nodes to uniform belief
reset_children_to_prior(Graph, Children) ->
    lists:foldl(fun(ChildId, GAcc) ->
        case dua_graph:get_node(GAcc, ChildId) of
            {ok, #dua_node{states = States} = Node} ->
                PriorBelief = dua_belief:uniform_belief(States),
                {ok, G1} = update_belief_direct(GAcc, ChildId, PriorBelief),
                G1;
            _ -> GAcc
        end
    end, Graph, Children).

%% @private: Filter the graph to retain only specified nodes and corresponding edges
filter_graph_nodes(#dua_graph{nodes = Nodes, edges = Edges} = Graph, KeepNodes) ->
    FilteredNodes = maps:with(KeepNodes, Nodes),
    FilteredEdges = lists:filter(fun(#dua_edge{from = From, to = To}) ->
        lists:member(From, KeepNodes) andalso lists:member(To, KeepNodes)
    end, Edges),
    Graph#dua_graph{nodes = FilteredNodes, edges = FilteredEdges}.

%% @private: Extract current node beliefs as a map
extract_current_beliefs(#dua_graph{nodes = Nodes}) ->
    maps:map(fun(_NodeId, #dua_node{current_belief = Belief}) -> Belief end, Nodes).

%% @private: Detect impossible evidence states (stub: returns [] here)
detect_direct_conflicts(_Graph, _Evidence) ->
    [].

%% @private: Detect very low-probability evidence (stub logic)
detect_probabilistic_conflicts(Graph, Evidence) ->
    case dua_forward:infer(Graph, Evidence) of
        {ok, #{confidence := Confidence}} when Confidence < 0.01 ->
            [#{conflict => low_probability, nodes => maps:keys(Evidence)}];
        _ -> []
    end.

%% @private: Weighted belief blending using interpolation
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
jeffrey_belief_revision(Graph, NewEvidence, _Options) ->
    propagate_evidence(Graph, NewEvidence).

%% @private: Minimal change principle (stub, uses direct propagation)
minimal_change_revision(Graph, NewEvidence, _Options) ->
    propagate_evidence(Graph, NewEvidence).

%% @private: Apply exponential decay to all node beliefs (toward uniform)
apply_temporal_decay(#dua_graph{nodes = Nodes} = Graph, DecayRate, TimeSteps) ->
    DecayFactor = math:pow(DecayRate, TimeSteps),
    UpdatedNodes = maps:map(fun(_NodeId, #dua_node{id = Nid, current_belief = Belief, states = States} = Node) ->
        UniformBelief = dua_belief:uniform_belief(States),
        DecayedBelief = dua_belief:interpolate(UniformBelief, Belief, DecayFactor),
        Node#dua_node{current_belief = DecayedBelief}
    end, Nodes),
    {ok, Graph#dua_graph{nodes = UpdatedNodes}}.

%% @private: Return a uniform weights list of size N
uniform_weights(N) ->
    W = 1.0 / N, lists:duplicate(N, W).

%% @private: Combine multiple evidence maps using weights (very simplified: merges only strongly weighted evidence)
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
update_children_beliefs(Graph, ParentId, Children) ->
    lists:foldl(fun(ChildId, GAcc) ->
        case update_child_belief(GAcc, ParentId, ChildId) of
            {ok, G1} -> G1;
            _ -> GAcc
        end
    end, Graph, Children).

%% @private: Directly update a child node using parent (stub: just interpolates a bit)
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

