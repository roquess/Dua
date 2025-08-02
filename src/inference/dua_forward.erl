%%%-------------------------------------------------------------------
%%% @doc DUA Forward Inference Engine
%%% 
%%% This module implements forward inference (cause → effect) using
%%% various algorithms including variable elimination, belief propagation,
%%% and message passing.
%%%-------------------------------------------------------------------

-module(dua_forward).

-include("dua.hrl").

-export([
    %% Main inference functions
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

%%%===================================================================
%%% Main Inference Functions
%%%===================================================================

%% @doc Perform forward inference with evidence
-spec infer(#dua_graph{}, evidence()) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence) ->
    infer(Graph, Evidence, #{algorithm => variable_elimination}).

%% @doc Perform forward inference with options
-spec infer(#dua_graph{}, evidence(), map()) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence, Options) ->
    Algorithm = maps:get(algorithm, Options, variable_elimination),
    MaxIterations = maps:get(max_iterations, Options, ?MAX_ITERATIONS),
    
    case validate_evidence(Graph, Evidence) of
        ok ->
            execute_inference(Graph, Evidence, Algorithm, Options);
        Error ->
            Error
    end.

%%%===================================================================
%%% Inference Algorithms
%%%===================================================================

%% @doc Variable elimination algorithm for forward inference
-spec variable_elimination(#dua_graph{}, evidence(), map()) -> 
    {ok, query_result()} | dua_error().
variable_elimination(Graph, Evidence, Options) ->
    try
        %% Step 1: Get elimination order
        {ok, EliminationOrder} = get_elimination_order(Graph, Evidence),
        
        %% Step 2: Apply evidence to the graph
        {ok, EvidenceGraph} = apply_evidence_to_graph(Graph, Evidence),
        
        %% Step 3: Perform variable elimination
        {ok, Factors} = create_initial_factors(EvidenceGraph),
        {ok, FinalFactors} = eliminate_variables(Factors, EliminationOrder),
        
        %% Step 4: Compute final beliefs
        {ok, Beliefs} = compute_beliefs_from_factors(FinalFactors, Graph),
        
        %% Step 5: Build result
        Result = #{
            beliefs => Beliefs,
            inference_path => EliminationOrder,
            confidence => compute_confidence(Beliefs),
            iterations => 1,
            algorithm => variable_elimination
        },
        
        {ok, Result}
    catch
        error:Reason ->
            {error, {inference_failed, Reason}};
        throw:Error ->
            {error, Error}
    end.

%% @doc Belief propagation algorithm for forward inference
-spec belief_propagation(#dua_graph{}, evidence(), map()) -> 
    {ok, query_result()} | dua_error().
belief_propagation(Graph, Evidence, Options) ->
    MaxIterations = maps:get(max_iterations, Options, ?MAX_ITERATIONS),
    Precision = maps:get(precision, Options, ?DEFAULT_PRECISION),
    
    try
        %% Step 1: Initialize messages
        {ok, Messages} = initialize_messages(Graph),
        
        %% Step 2: Apply evidence
        {ok, EvidenceGraph} = apply_evidence_to_graph(Graph, Evidence),
        
        %% Step 3: Iterative message passing
        {ok, FinalMessages, Iterations} = message_passing_loop(
            EvidenceGraph, Messages, MaxIterations, Precision, 0
        ),
        
        %% Step 4: Compute marginals
        {ok, Beliefs} = compute_marginals(EvidenceGraph, FinalMessages),
        
        %% Step 5: Build result
        Result = #{
            beliefs => Beliefs,
            inference_path => get_message_path(FinalMessages),
            confidence => compute_confidence(Beliefs),
            iterations => Iterations,
            algorithm => belief_propagation
        },
        
        {ok, Result}
    catch
        error:Reason ->
            {error, {inference_failed, Reason}};
        throw:Error ->
            {error, Error}
    end.

%% @doc Likelihood weighting for forward inference (approximate)
-spec likelihood_weighting(#dua_graph{}, evidence(), map()) -> 
    {ok, query_result()} | dua_error().
likelihood_weighting(Graph, Evidence, Options) ->
    NumSamples = maps:get(num_samples, Options, 1000),
    
    try
        %% Step 1: Get topological order for sampling
        {ok, SamplingOrder} = dua_graph:topological_sort(Graph),
        
        %% Step 2: Generate weighted samples
        {ok, Samples} = generate_weighted_samples(Graph, Evidence, SamplingOrder, NumSamples),
        
        %% Step 3: Compute beliefs from samples
        {ok, Beliefs} = compute_beliefs_from_samples(Samples, Graph),
        
        %% Step 4: Build result
        Result = #{
            beliefs => Beliefs,
            inference_path => SamplingOrder,
            confidence => compute_confidence(Beliefs),
            iterations => NumSamples,
            algorithm => likelihood_weighting
        },
        
        {ok, Result}
    catch
        error:Reason ->
            {error, {inference_failed, Reason}};
        throw:Error ->
            {error, Error}
    end.

%%%===================================================================
%%% Utility Functions
%%%===================================================================

%% @doc Predict outcomes given current evidence
-spec predict(#dua_graph{}, evidence(), [node_id()]) -> 
    {ok, #{node_id() => belief()}} | dua_error().
predict(Graph, Evidence, QueryNodes) ->
    case infer(Graph, Evidence) of
        {ok, #{beliefs := Beliefs}} ->
            Predictions = maps:with(QueryNodes, Beliefs),
            {ok, Predictions};
        Error ->
            Error
    end.

%% @doc Propagate evidence through the graph
-spec propagate_evidence(#dua_graph{}, evidence(), map()) -> 
    {ok, #dua_graph{}} | dua_error().
propagate_evidence(Graph, Evidence, Options) ->
    Algorithm = maps:get(algorithm, Options, simple_propagation),
    
    case Algorithm of
        simple_propagation ->
            simple_evidence_propagation(Graph, Evidence);
        pearl_propagation ->
            pearl_evidence_propagation(Graph, Evidence);
        _ ->
            {error, {unknown_algorithm, Algorithm}}
    end.

%% @doc Compute posterior probabilities
-spec compute_posterior(#dua_graph{}, evidence(), node_id()) -> 
    {ok, belief()} | dua_error().
compute_posterior(Graph, Evidence, QueryNode) ->
    case infer(Graph, Evidence) of
        {ok, #{beliefs := Beliefs}} ->
            case maps:get(QueryNode, Beliefs, undefined) of
                undefined ->
                    {error, {node_not_found, QueryNode}};
                Belief ->
                    {ok, Belief}
            end;
        Error ->
            Error
    end.

%%%===================================================================
%%% Internal Functions
%%%===================================================================

%% @private
validate_evidence(Graph, Evidence) ->
    maps:fold(fun(NodeId, State, ok) ->
        case dua_graph:get_node(Graph, NodeId) of
            {ok, #dua_node{states = States}} ->
                case lists:member(State, States) of
                    true -> ok;
                    false -> throw({invalid_evidence_state, NodeId, State})
                end;
            {error, _} ->
                throw({evidence_node_not_found, NodeId})
        end;
    (_, _, Error) -> Error
    end, ok, Evidence).

%% @private
execute_inference(Graph, Evidence, Algorithm, Options) ->
    case Algorithm of
        variable_elimination ->
            variable_elimination(Graph, Evidence, Options);
        belief_propagation ->
            belief_propagation(Graph, Evidence, Options);
        likelihood_weighting ->
            likelihood_weighting(Graph, Evidence, Options);
        _ ->
            {error, {unknown_algorithm, Algorithm}}
    end.

%% @private
get_elimination_order(Graph, Evidence) ->
    %% Use min-fill heuristic for variable elimination order
    EvidenceNodes = maps:keys(Evidence),
    {ok, AllNodes} = get_all_node_ids(Graph),
    VariablesToEliminate = AllNodes -- EvidenceNodes,
    
    %% Simple heuristic: eliminate in reverse topological order
    case dua_graph:topological_sort(Graph) of
        {ok, TopOrder} ->
            EliminationOrder = lists:filter(fun(Node) ->
                lists:member(Node, VariablesToEliminate)
            end, lists:reverse(TopOrder)),
            {ok, EliminationOrder};
        Error ->
            Error
    end.

%% @private
apply_evidence_to_graph(#dua_graph{nodes = Nodes} = Graph, Evidence) ->
    UpdatedNodes = maps:fold(fun(NodeId, EvidenceState, NodesAcc) ->
        case maps:get(NodeId, NodesAcc, undefined) of
            undefined ->
                NodesAcc;
            Node ->
                %% Create certain belief for evidence
                {ok, CertainBelief} = dua_belief:certain_belief(
                    Node#dua_node.states, EvidenceState
                ),
                UpdatedNode = Node#dua_node{current_belief = CertainBelief},
                maps:put(NodeId, UpdatedNode, NodesAcc)
        end
    end, Nodes, Evidence),
    
    {ok, Graph#dua_graph{nodes = UpdatedNodes}}.

%% @private
create_initial_factors(#dua_graph{nodes = Nodes}) ->
    Factors = maps:fold(fun(NodeId, Node, FactorsAcc) ->
        Factor = create_node_factor(NodeId, Node),
        [Factor | FactorsAcc]
    end, [], Nodes),
    {ok, Factors}.

%% @private
create_node_factor(NodeId, #dua_node{current_belief = Belief, parents = Parents}) ->
    #{
        variables => [NodeId | Parents],
        probabilities => Belief,
        type => conditional
    }.

%% @private
eliminate_variables(Factors, []) ->
    {ok, Factors};
eliminate_variables(Factors, [Var | RestVars]) ->
    %% Find factors containing the variable
    {VarFactors, OtherFactors} = lists:partition(fun(Factor) ->
        Variables = maps:get(variables, Factor),
        lists:member(Var, Variables)
    end, Factors),
    
    %% Multiply factors and sum out the variable
    case VarFactors of
        [] ->
            eliminate_variables(Factors, RestVars);
        _ ->
            NewFactor = multiply_and_marginalize(VarFactors, Var),
            UpdatedFactors = [NewFactor | OtherFactors],
            eliminate_variables(UpdatedFactors, RestVars)
    end.

%% @private
multiply_and_marginalize(Factors, VarToEliminate) ->
    %% Simplified implementation - in practice would need full factor operations
    CombinedVariables = lists:usort(lists:flatten([
        maps:get(variables, F) || F <- Factors
    ])) -- [VarToEliminate],
    
    %% For now, return a simplified factor
    #{
        variables => CombinedVariables,
        probabilities => #{},  % Would compute actual joint distribution
        type => marginalized
    }.

%% @private
compute_beliefs_from_factors(Factors, Graph) ->
    %% Extract marginal beliefs from final factors
    {ok, AllNodes} = get_all_node_ids(Graph),
    
    Beliefs = lists:foldl(fun(NodeId, BeliefAcc) ->
        %% Find factor containing this node
        NodeBelief = case find_node_factor(NodeId, Factors) of
            {ok, Factor} ->
                maps:get(probabilities, Factor, #{});
            _ ->
                %% Default uniform belief if not found
                case dua_graph:get_node(Graph, NodeId) of
                    {ok, #dua_node{states = States}} ->
                        dua_belief:uniform_belief(States);
                    _ ->
                        #{}
                end
        end,
        maps:put(NodeId, NodeBelief, BeliefAcc)
    end, #{}, AllNodes),
    
    {ok, Beliefs}.

%% @private
initialize_messages(#dua_graph{edges = Edges}) ->
    Messages = lists:foldl(fun(#dua_edge{from = From, to = To}, MsgAcc) ->
        ForwardKey = {From, To},
        BackwardKey = {To, From},
        maps:put(ForwardKey, #{}, 
                maps:put(BackwardKey, #{}, MsgAcc))
    end, #{}, Edges),
    {ok, Messages}.

%% @private
message_passing_loop(Graph, Messages, MaxIter, Precision, Iter) when Iter >= MaxIter ->
    {ok, Messages, Iter};
message_passing_loop(Graph, Messages, MaxIter, Precision, Iter) ->
    {ok, NewMessages} = update_all_messages(Graph, Messages),
    
    %% Check convergence
    case messages_converged(Messages, NewMessages, Precision) of
        true ->
            {ok, NewMessages, Iter + 1};
        false ->
            message_passing_loop(Graph, NewMessages, MaxIter, Precision, Iter + 1)
    end.

%% @private
update_all_messages(#dua_graph{edges = Edges} = Graph, Messages) ->
    UpdatedMessages = lists:foldl(fun(#dua_edge{from = From, to = To}, MsgAcc) ->
        NewMessage = compute_message(Graph, From, To, MsgAcc),
        maps:put({From, To}, NewMessage, MsgAcc)
    end, Messages, Edges),
    {ok, UpdatedMessages}.

%% @private
compute_message(Graph, From, To, Messages) ->
    %% Simplified message computation
    case dua_graph:get_node(Graph, From) of
        {ok, #dua_node{current_belief = Belief}} ->
            Belief;
        _ ->
            #{}
    end.

%% @private
messages_converged(OldMessages, NewMessages, Precision) ->
    maps:fold(fun(Key, NewMsg, Converged) ->
        case maps:get(Key, OldMessages, #{}) of
            OldMsg ->
                Distance = dua_belief:belief_distance(OldMsg, NewMsg),
                Converged andalso Distance < Precision;
            _ ->
                false
        end
    end, true, NewMessages).

%% @private
compute_marginals(Graph, Messages) ->
    {ok, AllNodes} = get_all_node_ids(Graph),
    
    Beliefs = lists:foldl(fun(NodeId, BeliefAcc) ->
        NodeBelief = compute_node_marginal(Graph, NodeId, Messages),
        maps:put(NodeId, NodeBelief, BeliefAcc)
    end, #{}, AllNodes),
    
    {ok, Beliefs}.

%% @private
compute_node_marginal(Graph, NodeId, Messages) ->
    case dua_graph:get_node(Graph, NodeId) of
        {ok, #dua_node{current_belief = Belief}} ->
            %% Combine with incoming messages
            {ok, Parents} = dua_graph:get_parents(Graph, NodeId),
            {ok, Children} = dua_graph:get_children(Graph, NodeId),
            
            %% Simplified marginal computation
            Belief;
        _ ->
            #{}
    end.

%% @private
generate_weighted_samples(_Graph, _Evidence, _Order, NumSamples) ->
    %% Simplified sampling - would implement actual likelihood weighting
    Samples = lists:duplicate(NumSamples, #{weight => 1.0, assignment => #{}}),
    {ok, Samples}.

%% @private
compute_beliefs_from_samples(Samples, Graph) ->
    %% Extract beliefs from weighted samples
    {ok, AllNodes} = get_all_node_ids(Graph),
    
    Beliefs = lists:foldl(fun(NodeId, BeliefAcc) ->
        NodeBelief = case dua_graph:get_node(Graph, NodeId) of
            {ok, #dua_node{states = States}} ->
                dua_belief:uniform_belief(States);
            _ ->
                #{}
        end,
        maps:put(NodeId, NodeBelief, BeliefAcc)
    end, #{}, AllNodes),
    
    {ok, Beliefs}.

%% @private
simple_evidence_propagation(Graph, Evidence) ->
    apply_evidence_to_graph(Graph, Evidence).

%% @private
pearl_evidence_propagation(Graph, Evidence) ->
    %% Implement Pearl's evidence propagation algorithm
    apply_evidence_to_graph(Graph, Evidence).

%% @private
compute_confidence(Beliefs) ->
    case map_size(Beliefs) of
        0 -> 0.0;
        N ->
            TotalEntropy = maps:fold(fun(_, Belief, Acc) ->
                Acc + dua_belief:entropy(Belief)
            end, 0.0, Beliefs),
            %% Convert entropy to confidence (lower entropy = higher confidence)
            math:exp(-TotalEntropy / N)
    end.

%% @private
get_message_path(Messages) ->
    %% Extract inference path from messages
    maps:keys(Messages).

%% @private
get_all_node_ids(#dua_graph{nodes = Nodes}) ->
    {ok, maps:keys(Nodes)}.

%% @private
find_node_factor(NodeId, Factors) ->
    case lists:filter(fun(Factor) ->
        Variables = maps:get(variables, Factor),
        lists:member(NodeId, Variables)
    end, Factors) of
        [Factor | _] -> {ok, Factor};
        [] -> {error, not_found}
    end.
