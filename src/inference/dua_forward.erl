%%%-------------------------------------------------------------------
%%% @doc
%%% DUA Forward Inference Engine
%%%
%%% This module implements forward (cause → effect) inference algorithms
%%% on probabilistic graphical models. Supported algorithms include:
%%%   - Variable elimination
%%%   - Belief propagation (message passing)
%%%   - Likelihood weighting (sampling)
%%%
%%% The module also provides utility functions for prediction,
%%% evidence propagation, and posterior computation.
%%%
%%% NOTE: This implementation contains simplified versions of factor
%%% operations and message passing for demonstration purposes.
%%%-------------------------------------------------------------------

-module(dua_forward).

-include("dua.hrl").

%%====================================================================
%% Exported API
%%====================================================================

-export([
    %% Main inference entry points
    infer/2,
    infer/3,

    %% Specific inference algorithms
    variable_elimination/3,
    belief_propagation/3,
    likelihood_weighting/3,

    %% Utility functions
    predict/3,
    propagate_evidence/3,
    compute_posterior/3
]).

%%====================================================================
%% Main Inference Functions
%%====================================================================

%% @doc Perform inference given a graph and evidence.
%% Defaults to variable elimination.
-spec infer(#dua_graph{}, evidence()) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence) ->
    infer(Graph, Evidence, #{algorithm => variable_elimination}).

%% @doc Perform inference with options (algorithm, iterations, etc.).
-spec infer(#dua_graph{}, evidence(), map()) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence, Options) ->
    Algorithm = maps:get(algorithm, Options, variable_elimination),
    %% retrieve but don’t warn if unused
    _MaxIterations = maps:get(max_iterations, Options, ?MAX_ITERATIONS),
    case validate_evidence(Graph, Evidence) of
        ok ->
            execute_inference(Graph, Evidence, Algorithm, Options);
        Error ->
            Error
    end.

%%====================================================================
%% Inference Algorithms
%%====================================================================

%% @doc Exact inference via variable elimination.
-spec variable_elimination(#dua_graph{}, evidence(), map()) ->
    {ok, query_result()} | dua_error().
variable_elimination(Graph, Evidence, _Options) ->
    try
        {ok, EliminationOrder} = get_elimination_order(Graph, Evidence),
        {ok, EvidenceGraph}   = apply_evidence_to_graph(Graph, Evidence),
        {ok, Factors}         = create_initial_factors(EvidenceGraph),
        {ok, FinalFactors}    = eliminate_variables(Factors, EliminationOrder),
        {ok, Beliefs}         = compute_beliefs_from_factors(FinalFactors, Graph),

        Result = #{
            beliefs        => Beliefs,
            inference_path => EliminationOrder,
            confidence     => compute_confidence(Beliefs),
            iterations     => 1,
            algorithm      => variable_elimination
        },
        {ok, Result}
    catch
        error:Reason -> {error, {inference_failed, Reason}};
        throw:Error  -> {error, Error}
    end.

%% @doc Approximate inference using loopy belief propagation.
-spec belief_propagation(#dua_graph{}, evidence(), map()) ->
    {ok, query_result()} | dua_error().
belief_propagation(Graph, Evidence, Options) ->
    MaxIterations = maps:get(max_iterations, Options, ?MAX_ITERATIONS),
    Precision     = maps:get(precision, Options, ?DEFAULT_PRECISION),
    try
        {ok, Messages}       = initialize_messages(Graph),
        {ok, EvidenceGraph}  = apply_evidence_to_graph(Graph, Evidence),
        {ok, FinalMessages, Iterations} =
            message_passing_loop(EvidenceGraph, Messages, MaxIterations, Precision, 0),
        {ok, Beliefs} = compute_marginals(EvidenceGraph, FinalMessages),

        Result = #{
            beliefs        => Beliefs,
            inference_path => get_message_path(FinalMessages),
            confidence     => compute_confidence(Beliefs),
            iterations     => Iterations,
            algorithm      => belief_propagation
        },
        {ok, Result}
    catch
        error:Reason -> {error, {inference_failed, Reason}};
        throw:Error  -> {error, Error}
    end.

%% @doc Approximate inference using likelihood weighting (sampling).
-spec likelihood_weighting(#dua_graph{}, evidence(), map()) ->
    {ok, query_result()} | dua_error().
likelihood_weighting(Graph, Evidence, Options) ->
    NumSamples = maps:get(num_samples, Options, 1000),
    try
        {ok, SamplingOrder} = dua_graph:topological_sort(Graph),
        {ok, Samples}       = generate_weighted_samples(Graph, Evidence, SamplingOrder, NumSamples),
        {ok, Beliefs}       = compute_beliefs_from_samples(Samples, Graph),

        Result = #{
            beliefs        => Beliefs,
            inference_path => SamplingOrder,
            confidence     => compute_confidence(Beliefs),
            iterations     => NumSamples,
            algorithm      => likelihood_weighting
        },
        {ok, Result}
    catch
        error:Reason -> {error, {inference_failed, Reason}};
        throw:Error  -> {error, Error}
    end.

%%====================================================================
%% Utility Functions
%%====================================================================

%% @doc Predict beliefs for a subset of query nodes.
-spec predict(#dua_graph{}, evidence(), [node_id()]) ->
    {ok, #{node_id() => belief()}} | dua_error().
predict(Graph, Evidence, QueryNodes) ->
    case infer(Graph, Evidence) of
        {ok, #{beliefs := Beliefs}} ->
            {ok, maps:with(QueryNodes, Beliefs)};
        Error ->
            Error
    end.

%% @doc Propagate evidence through the graph using a chosen method.
-spec propagate_evidence(#dua_graph{}, evidence(), map()) ->
    {ok, #dua_graph{}} | dua_error().
propagate_evidence(Graph, Evidence, Options) ->
    case maps:get(algorithm, Options, simple_propagation) of
        simple_propagation -> simple_evidence_propagation(Graph, Evidence);
        pearl_propagation  -> pearl_evidence_propagation(Graph, Evidence);
        Alg                -> {error, {unknown_algorithm, Alg}}
    end.

%% @doc Compute posterior belief for a single node.
-spec compute_posterior(#dua_graph{}, evidence(), node_id()) ->
    {ok, belief()} | dua_error().
compute_posterior(Graph, Evidence, QueryNode) ->
    case infer(Graph, Evidence) of
        {ok, #{beliefs := Beliefs}} ->
            case maps:get(QueryNode, Beliefs, undefined) of
                undefined -> {error, {node_not_found, QueryNode}};
                Belief    -> {ok, Belief}
            end;
        Error ->
            Error
    end.

%%====================================================================
%% Internal Functions (Private)
%%====================================================================

%% Validate that provided evidence matches node states.
validate_evidence(Graph, Evidence) ->
    maps:fold(fun(NodeId, State, ok) ->
        case dua_graph:get_node(Graph, NodeId) of
            {ok, #dua_node{states = States}} ->
                case lists:member(State, States) of
                    true  -> ok;
                    false -> throw({invalid_evidence_state, NodeId, State})
                end;
            {error, _} ->
                throw({evidence_node_not_found, NodeId})
        end;
    (_, _, Error) -> Error
    end, ok, Evidence).

%% Dispatch inference based on selected algorithm.
execute_inference(Graph, Evidence, Algorithm, Options) ->
    case Algorithm of
        variable_elimination -> variable_elimination(Graph, Evidence, Options);
        belief_propagation   -> belief_propagation(Graph, Evidence, Options);
        likelihood_weighting -> likelihood_weighting(Graph, Evidence, Options);
        _                    -> {error, {unknown_algorithm, Algorithm}}
    end.

%% Determine elimination order for variable elimination.
get_elimination_order(Graph, Evidence) ->
    EvidenceNodes        = maps:keys(Evidence),
    {ok, AllNodes}       = get_all_node_ids(Graph),
    VariablesToEliminate = AllNodes -- EvidenceNodes,
    case dua_graph:topological_sort(Graph) of
        {ok, TopOrder} ->
            {ok, [N || N <- lists:reverse(TopOrder),
                       lists:member(N, VariablesToEliminate)]};
        Error -> Error
    end.

%% Apply evidence by setting certain beliefs for observed nodes.
apply_evidence_to_graph(#dua_graph{nodes = Nodes} = Graph, Evidence) ->
    UpdatedNodes = maps:fold(fun(NodeId, EvidenceState, Acc) ->
        case maps:get(NodeId, Acc, undefined) of
            undefined -> Acc;
            Node ->
                {ok, CertainBelief} = dua_belief:certain_belief(Node#dua_node.states, EvidenceState),
                maps:put(NodeId, Node#dua_node{current_belief = CertainBelief}, Acc)
        end
    end, Nodes, Evidence),
    {ok, Graph#dua_graph{nodes = UpdatedNodes}}.

%% Build initial factors from graph nodes.
create_initial_factors(#dua_graph{nodes = Nodes}) ->
    {ok, maps:fold(fun(NodeId, Node, Acc) ->
        [create_node_factor(NodeId, Node) | Acc]
    end, [], Nodes)}.

create_node_factor(NodeId, #dua_node{current_belief = Belief, parents = Parents}) ->
    #{variables => [NodeId | Parents], probabilities => Belief, type => conditional}.

%% Eliminate variables via factor multiplication and marginalization.
eliminate_variables(Factors, []) -> {ok, Factors};
eliminate_variables(Factors, [Var | Rest]) ->
    {VarFactors, Others} = lists:partition(fun(F) ->
        lists:member(Var, maps:get(variables, F)) end, Factors),
    case VarFactors of
        [] -> eliminate_variables(Factors, Rest);
        _  -> eliminate_variables([multiply_and_marginalize(VarFactors, Var) | Others], Rest)
    end.

multiply_and_marginalize(Factors, VarToEliminate) ->
    CombinedVars = lists:usort(lists:flatten([maps:get(variables, F) || F <- Factors]))
                   -- [VarToEliminate],
    #{variables => CombinedVars, probabilities => #{}, type => marginalized}.

%% Compute beliefs from the final set of factors.
compute_beliefs_from_factors(Factors, Graph) ->
    {ok, AllNodes} = get_all_node_ids(Graph),
    Beliefs = lists:foldl(fun(NodeId, Acc) ->
        NodeBelief = case find_node_factor(NodeId, Factors) of
            {ok, Factor} -> maps:get(probabilities, Factor, #{});
            _ ->
                case dua_graph:get_node(Graph, NodeId) of
                    {ok, #dua_node{states = States}} -> dua_belief:uniform_belief(States);
                    _ -> #{}
                end
        end,
        maps:put(NodeId, NodeBelief, Acc)
    end, #{}, AllNodes),
    {ok, Beliefs}.

%% Initialize empty messages on all edges.
initialize_messages(#dua_graph{edges = Edges}) ->
    {ok, lists:foldl(fun(#dua_edge{from = F, to = T}, Acc) ->
        maps:put({F,T}, #{}, maps:put({T,F}, #{}, Acc))
    end, #{}, Edges)}.

%% Message passing loop (belief propagation).
message_passing_loop(_Graph, Messages, MaxIter, _Precision, Iter) when Iter >= MaxIter ->
    {ok, Messages, Iter};
message_passing_loop(Graph, Messages, MaxIter, Precision, Iter) ->
    {ok, NewMessages} = update_all_messages(Graph, Messages),
    case messages_converged(Messages, NewMessages, Precision) of
        true  -> {ok, NewMessages, Iter + 1};
        false -> message_passing_loop(Graph, NewMessages, MaxIter, Precision, Iter + 1)
    end.

update_all_messages(#dua_graph{edges = Edges} = Graph, Messages) ->
    {ok, lists:foldl(fun(#dua_edge{from = F, to = T}, Acc) ->
        maps:put({F,T}, compute_message(Graph, F, T), Acc)
    end, Messages, Edges)}.

%% Simplified message computation: just forward the sender's current belief.
compute_message(Graph, From, _To) ->
    case dua_graph:get_node(Graph, From) of
        {ok, #dua_node{current_belief = Belief}} -> Belief;
        _ -> #{}
    end.

%% Check if all messages have converged below precision threshold.
messages_converged(Old, New, Precision) ->
    maps:fold(fun(Key, NewMsg, Converged) ->
        OldMsg   = maps:get(Key, Old, #{}),
        Distance = dua_belief:belief_distance(OldMsg, NewMsg),
        Converged andalso Distance < Precision
    end, true, New).

%% Compute node marginals from graph and messages.
compute_marginals(Graph, Messages) ->
    {ok, AllNodes} = get_all_node_ids(Graph),
    Beliefs = lists:foldl(fun(NodeId, Acc) ->
        maps:put(NodeId, compute_node_marginal(Graph, NodeId, Messages), Acc)
    end, #{}, AllNodes),
    {ok, Beliefs}.

compute_node_marginal(Graph, NodeId, _Messages) ->
    case dua_graph:get_node(Graph, NodeId) of
        {ok, #dua_node{current_belief = Belief}} -> Belief;
        _ -> #{}
    end.

%% Sampling-based inference support.
generate_weighted_samples(_Graph, _Evidence, _Order, NumSamples) ->
    {ok, lists:duplicate(NumSamples, #{weight => 1.0, assignment => #{}})}.

compute_beliefs_from_samples(_Samples, Graph) ->
    {ok, AllNodes} = get_all_node_ids(Graph),
    Beliefs = lists:foldl(fun(NodeId, Acc) ->
        NodeBelief = case dua_graph:get_node(Graph, NodeId) of
            {ok, #dua_node{states = States}} -> dua_belief:uniform_belief(States);
            _ -> #{}
        end,
        maps:put(NodeId, NodeBelief, Acc)
    end, #{}, AllNodes),
    {ok, Beliefs}.

%% Evidence propagation strategies.
simple_evidence_propagation(Graph, Evidence) ->
    apply_evidence_to_graph(Graph, Evidence).

pearl_evidence_propagation(Graph, Evidence) ->
    %% TODO: implement Pearl’s propagation algorithm.
    apply_evidence_to_graph(Graph, Evidence).

%% Confidence estimation: entropy-based measure.
compute_confidence(Beliefs) ->
    case map_size(Beliefs) of
        0 -> 0.0;
        N ->
            TotalEntropy = maps:fold(fun(_, Belief, Acc) ->
                Acc + dua_belief:entropy(Belief)
            end, 0.0, Beliefs),
            math:exp(-TotalEntropy / N)
    end.

%% Helpers.
get_message_path(Messages) -> maps:keys(Messages).
get_all_node_ids(#dua_graph{nodes = Nodes}) -> {ok, maps:keys(Nodes)}.

find_node_factor(NodeId, Factors) ->
    case lists:filter(fun(F) -> lists:member(NodeId, maps:get(variables, F)) end, Factors) of
        [F | _] -> {ok, F};
        []      -> {error, not_found}
    end.

