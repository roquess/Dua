%%%-------------------------------------------------------------------
%%% @doc DUA Backward Inference Engine
%%%
%%% This module implements backward inference (effect → cause) using
%%% Bayesian inversion, abductive reasoning, and causal analysis.
%%%-------------------------------------------------------------------
-module(dua_backward).
-include("dua.hrl").

%% API
-export([
    %% Main inference functions
    infer/2,
    infer/3,
    %% Specific inference algorithms
    bayesian_inversion/2,  %% Removed unused 'Options' parameter
    abductive_reasoning/2,  %% Removed unused 'Options' parameter
    most_probable_explanation/2,  %% Removed unused 'Options' parameter
    %% Utility functions
    explain/2,
    find_causes/2,
    causal_strength/3
]).

%%%===================================================================
%%% Main Inference Functions
%%%===================================================================

%% @doc Perform backward inference with evidence.
%% @param Graph The DUA graph.
%% @param Evidence The observed evidence.
%% @return {ok, query_result()} if successful, otherwise dua_error().
-spec infer(#dua_graph{}, #{node_id() => node_state()}) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence) ->
    infer(Graph, Evidence, #{algorithm => bayesian_inversion}).

%% @doc Perform backward inference with options.
%% @param Options Map of options (e.g., algorithm, max_iterations).
%% @return {ok, query_result()} if successful, otherwise dua_error().
-spec infer(#dua_graph{}, #{node_id() => node_state()}, map()) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence, _Options) ->  %% '_Options' intentionally unused for now
    Algorithm = maps:get(algorithm, _Options, bayesian_inversion),
    case validate_evidence(Graph, Evidence) of
        ok ->
            execute_backward_inference(Graph, Evidence, Algorithm, _Options);
        Error ->
            Error
    end.

%%%===================================================================
%%% Inference Algorithms
%%%===================================================================

%% @doc Bayesian inversion for backward inference.
%% @param Graph The DUA graph.
%% @param Evidence The observed evidence.
%% @return {ok, query_result()} if successful, otherwise dua_error().
-spec bayesian_inversion(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, query_result()} | dua_error().
bayesian_inversion(Graph, Evidence) ->
    try
        %% Step 1: Identify potential causes (ancestors of evidence nodes)
        {ok, CauseNodes} = identify_potential_causes(Graph, Evidence),
        %% Step 2: For each potential cause, compute P(cause|evidence)
        {ok, CausalBeliefs} = compute_causal_beliefs(Graph, Evidence, CauseNodes),
        %% Step 3: Rank causes by posterior probability
        {ok, RankedCauses} = rank_causes_by_probability(CausalBeliefs),
        %% Step 4: Build result
        Result = #{
            beliefs => CausalBeliefs,
            inference_path => RankedCauses,
            confidence => compute_backward_confidence(CausalBeliefs),
            iterations => 1,
            algorithm => bayesian_inversion,
            causal_ranking => RankedCauses
        },
        {ok, Result}
    catch
        error:Reason ->
            {error, {backward_inference_failed, Reason}};
        throw:Error ->
            {error, Error}
    end.

%% @doc Abductive reasoning for finding best explanations.
%% @param Graph The DUA graph.
%% @param Evidence The observed evidence.
%% @return {ok, query_result()} if successful, otherwise dua_error().
-spec abductive_reasoning(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, query_result()} | dua_error().
abductive_reasoning(Graph, Evidence) ->
    MaxExplanations = 10,  %% Default value, previously unused 'Options'
    try
        %% Step 1: Generate possible explanations
        {ok, Explanations} = generate_explanations(Graph, Evidence, MaxExplanations),
        %% Step 2: Score each explanation
        {ok, ScoredExplanations} = score_explanations(Graph, Evidence, Explanations),
        %% Step 3: Select best explanation(s)
        {ok, BestExplanations} = select_best_explanations(ScoredExplanations),
        %% Step 4: Convert to belief format
        {ok, ExplanationBeliefs} = explanations_to_beliefs(Graph, BestExplanations),
        %% Step 5: Build result
        Result = #{
            beliefs => ExplanationBeliefs,
            inference_path => extract_explanation_path(BestExplanations),
            confidence => compute_explanation_confidence(ScoredExplanations),
            iterations => length(Explanations),
            algorithm => abductive_reasoning,
            explanations => BestExplanations
        },
        {ok, Result}
    catch
        error:Reason ->
            {error, {abductive_reasoning_failed, Reason}};
        throw:Error ->
            {error, Error}
    end.

%% @doc Most Probable Explanation (MPE) algorithm.
%% @param Graph The DUA graph.
%% @param Evidence The observed evidence.
%% @return {ok, query_result()} if successful, otherwise dua_error().
-spec most_probable_explanation(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, query_result()} | dua_error().
most_probable_explanation(Graph, Evidence) ->
    try
        %% Step 1: Set up optimization problem
        {ok, Variables} = get_explanation_variables(Graph, Evidence),
        %% Step 2: Use dynamic programming or branch-and-bound
        {ok, OptimalAssignment} = find_optimal_assignment(Graph, Evidence, Variables),
        %% Step 3: Compute probability of optimal assignment
        {ok, OptimalProbability} = compute_assignment_probability(Graph, OptimalAssignment),
        %% Step 4: Convert assignment to beliefs
        {ok, MPEBeliefs} = assignment_to_beliefs(Graph, OptimalAssignment),
        %% Step 5: Build result
        Result = #{
            beliefs => MPEBeliefs,
            inference_path => Variables,
            confidence => OptimalProbability,
            iterations => 1,
            algorithm => most_probable_explanation,
            optimal_assignment => OptimalAssignment
        },
        {ok, Result}
    catch
        error:Reason ->
            {error, {mpe_failed, Reason}};
        throw:Error ->
            {error, Error}
    end.

%%%===================================================================
%%% Utility Functions
%%%===================================================================

%% @doc Explain observed effects by finding likely causes.
%% @param Graph The DUA graph.
%% @param Evidence The observed evidence.
%% @return {ok, #{explanation => term(), confidence => float()}} if successful, otherwise dua_error().
-spec explain(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, #{explanation => term(), confidence => float()}} | dua_error().
explain(Graph, Evidence) ->
    case infer(Graph, Evidence, #{algorithm => abductive_reasoning}) of
        {ok, #{explanations := Explanations, confidence := Confidence}} ->
            BestExplanation = case Explanations of
                [Best | _] -> Best;
                [] -> no_explanation_found
            end,
            {ok, #{explanation => BestExplanation, confidence => Confidence}};
        Error ->
            Error
    end.

%% @doc Find potential causes for given effects.
%% @param Graph The DUA graph.
%% @param Evidence The observed evidence.
%% @return {ok, [node_id()]} if successful, otherwise dua_error().
-spec find_causes(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, [node_id()]} | dua_error().
find_causes(Graph, Evidence) ->
    Threshold = 0.1,  %% Default value, previously unused 'Options'
    case infer(Graph, Evidence) of
        {ok, #{beliefs := Beliefs}} ->
            Causes = maps:fold(fun(NodeId, Belief, CauseAcc) ->
                case is_evidence_node(NodeId, Evidence) of
                    true -> CauseAcc;  %% Skip evidence nodes
                    false ->
                        case dua_belief:most_likely_state(Belief) of
                            {ok, {_State, Prob}} when Prob > Threshold ->
                                [NodeId | CauseAcc];
                            _ ->
                                CauseAcc
                        end
                end
            end, [], Beliefs),
            {ok, lists:reverse(Causes)};
        Error ->
            Error
    end.

%% @doc Calculate causal strength between cause and effect.
%% @param Graph The DUA graph.
%% @param CauseNode The potential cause node.
%% @param EffectNode The effect node.
%% @return {ok, float()} if successful, otherwise dua_error().
-spec causal_strength(#dua_graph{}, node_id(), node_id()) ->
    {ok, float()} | dua_error().
causal_strength(Graph, CauseNode, EffectNode) ->
    try
        %% Compute P(effect|cause=true)
        CauseTrue = maps:put(CauseNode, true, #{}),
        {ok, #{beliefs := BeliefsTrue}} = dua_forward:infer(Graph, CauseTrue),

        %% Compute P(effect|cause=false)
        CauseFalse = maps:put(CauseNode, false, #{}),
        {ok, #{beliefs := BeliefsFalse}} = dua_forward:infer(Graph, CauseFalse),

        %% Get beliefs for the effect node in both scenarios
        BeliefTrue = maps:get(EffectNode, BeliefsTrue, #{}),
        BeliefFalse = maps:get(EffectNode, BeliefsFalse, #{}),

        %% Calculate probabilities for most likely states
        case {dua_belief:most_likely_state(BeliefTrue), dua_belief:most_likely_state(BeliefFalse)} of
            {{ok, {_StateTrue, ProbTrue}}, {ok, {_StateFalse, ProbFalse}}} ->
                Strength = abs(ProbTrue - ProbFalse),
                {ok, Strength};
            _ ->
                {error, {cannot_compute_causal_strength, CauseNode, EffectNode}}
        end
    catch
        error:Reason ->
            {error, {causal_strength_failed, Reason}}
    end.

%%%===================================================================
%%% Internal Functions
%%%===================================================================

%% @private Validate evidence consistency.
-spec validate_evidence(#dua_graph{}, #{node_id() => node_state()}) -> ok | dua_error().
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

%% @private Execute backward inference using the chosen algorithm.
-spec execute_backward_inference(#dua_graph{}, #{node_id() => node_state()}, atom(), map()) ->
    {ok, query_result()} | dua_error().
execute_backward_inference(Graph, Evidence, Algorithm, _Options) ->
    case Algorithm of
        bayesian_inversion ->
            bayesian_inversion(Graph, Evidence);
        abductive_reasoning ->
            abductive_reasoning(Graph, Evidence);
        most_probable_explanation ->
            most_probable_explanation(Graph, Evidence);
        _ ->
            {error, {unknown_algorithm, Algorithm}}
    end.

%% @private Identify potential causes of evidence.
-spec identify_potential_causes(#dua_graph{}, #{node_id() => node_state()}) -> {ok, [node_id()]} | dua_error().
identify_potential_causes(Graph, Evidence) ->
    EvidenceNodes = maps:keys(Evidence),
    %% Find all ancestors of evidence nodes
    AllCauses = lists:foldl(fun(EvidenceNode, CauseAcc) ->
        case dua_graph:get_ancestors(Graph, EvidenceNode) of
            {ok, Ancestors} ->
                lists:usort(Ancestors ++ CauseAcc);
            _ ->
                CauseAcc
        end
    end, [], EvidenceNodes),
    {ok, AllCauses}.

%% @private Compute causal beliefs for potential causes.
-spec compute_causal_beliefs(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) ->
    {ok, #{node_id() => belief()}} | dua_error().
compute_causal_beliefs(Graph, Evidence, CauseNodes) ->
    %% For each potential cause, compute P(cause|evidence) using Bayes' rule
    CausalBeliefs = lists:foldl(fun(CauseNode, BeliefAcc) ->
        case compute_posterior_belief(Graph, CauseNode, Evidence) of
            {ok, Belief} ->
                maps:put(CauseNode, Belief, BeliefAcc);
            _ ->
                %% Use prior if posterior computation fails
                maps:put(CauseNode, get_prior_belief(Graph, CauseNode), BeliefAcc)
        end
    end, #{}, CauseNodes),
    {ok, CausalBeliefs}.

%% @private Compute posterior belief for a cause node.
-spec compute_posterior_belief(#dua_graph{}, node_id(), #{node_id() => node_state()}) ->
    {ok, belief()} | dua_error().
compute_posterior_belief(Graph, CauseNode, Evidence) ->
    try
        %% Use forward inference to compute P(evidence|cause) for each cause state
        case dua_graph:get_node(Graph, CauseNode) of
            {ok, #dua_node{states = States}} ->
                %% Compute likelihood for each state of the cause
                Likelihoods = lists:foldl(fun(State, LikelihoodAcc) ->
                    CauseEvidence = maps:put(CauseNode, State, Evidence),
                    case dua_forward:infer(Graph, CauseEvidence) of
                        {ok, #{beliefs := Beliefs}} ->
                            Likelihood = compute_evidence_likelihood(Beliefs, Evidence),
                            maps:put(State, Likelihood, LikelihoodAcc);
                        _ ->
                            maps:put(State, 0.0, LikelihoodAcc)
                    end
                end, #{}, States),
                %% Get prior probabilities
                Prior = get_prior_belief(Graph, CauseNode),
                %% Apply Bayes' rule: P(cause|evidence) ∝ P(evidence|cause) * P(cause)
                Posterior = maps:fold(fun(State, Likelihood, PostAcc) ->
                    PriorProb = maps:get(State, Prior, 0.0),
                    PostProb = Likelihood * PriorProb,
                    maps:put(State, PostProb, PostAcc)
                end, #{}, Likelihoods),
                %% Normalize
                NormalizedPosterior = dua_belief:normalize(Posterior),
                {ok, NormalizedPosterior};
            Error ->
                Error
        end
    catch
        error:Reason ->
            {error, Reason}
    end.

%% @private Get prior belief for a node.
-spec get_prior_belief(#dua_graph{}, node_id()) -> belief().
get_prior_belief(Graph, NodeId) ->
    case dua_graph:get_node(Graph, NodeId) of
        {ok, #dua_node{current_belief = Belief}} when Belief =/= undefined ->
            Belief;
        {ok, #dua_node{states = States}} ->
            dua_belief:uniform_belief(States);
        _ ->
            #{}
    end.

%% @private Compute likelihood of evidence given beliefs.
-spec compute_evidence_likelihood(#{node_id() => belief()}, #{node_id() => node_state()}) -> float().
compute_evidence_likelihood(Beliefs, Evidence) ->
    %% Compute how well the beliefs match the evidence
    maps:fold(
        fun(EvidenceNode, EvidenceState, LikelihoodAcc) ->
            NodeBelief = maps:get(EvidenceNode, Beliefs, #{}),
            %% If the node is not in Beliefs, treat as zero probability
            case maps:is_key(EvidenceNode, Beliefs) of
                true ->
                    StateProb = maps:get(EvidenceState, NodeBelief, 0.0),
                    LikelihoodAcc * StateProb;
                false ->
                    %% Node not found in beliefs - evidence is impossible
                    0.0
            end
        end,
        1.0,
        Evidence
    ).

%% @private Rank causes by probability.
-spec rank_causes_by_probability(#{node_id() => belief()}) -> {ok, [node_id()]} | dua_error().
rank_causes_by_probability(CausalBeliefs) ->
    %% Extract most likely state for each cause and sort by probability
    CauseProbs = maps:fold(fun(CauseNode, Belief, ProbAcc) ->
        case dua_belief:most_likely_state(Belief) of
            {ok, {_State, Prob}} ->
                [{CauseNode, Prob} | ProbAcc];
            _ ->
                [{CauseNode, 0.0} | ProbAcc]
        end
    end, [], CausalBeliefs),
    %% Sort by probability (descending)
    SortedCauses = lists:sort(fun({_, P1}, {_, P2}) -> P1 >= P2 end, CauseProbs),
    RankedCauses = [Node || {Node, _} <- SortedCauses],
    {ok, RankedCauses}.

%% @private Generate possible explanations for evidence.
-spec generate_explanations(#dua_graph{}, #{node_id() => node_state()}, pos_integer()) ->
    {ok, [[node_id()]]} | dua_error().
generate_explanations(Graph, Evidence, MaxExplanations) ->
    %% Generate possible explanations by exploring combinations of causes
    {ok, PotentialCauses} = identify_potential_causes(Graph, Evidence),
    %% Generate combinations of causes (simplified - could use more sophisticated search)
    Explanations = generate_cause_combinations(PotentialCauses, MaxExplanations),
    {ok, Explanations}.

%% @private Generate combinations of causes.
-spec generate_cause_combinations([node_id()], pos_integer()) -> [[node_id()]].
generate_cause_combinations(Causes, MaxCombinations) ->
    %% Generate all possible combinations up to MaxCombinations
    AllCombinations = generate_all_combinations(Causes),
    lists:sublist(AllCombinations, MaxCombinations).

%% @private Generate all possible combinations of causes.
-spec generate_all_combinations([node_id()]) -> [[node_id()]].
generate_all_combinations([]) ->
    [[]];
generate_all_combinations([H | T]) ->
    RestCombinations = generate_all_combinations(T),
    WithH = [[H | Combo] || Combo <- RestCombinations],
    WithoutH = RestCombinations,
    WithH ++ WithoutH.

%% @private Score explanations based on evidence.
-spec score_explanations(#dua_graph{}, #{node_id() => node_state()}, [[node_id()]]) ->
    {ok, [{[node_id()], float()}]} | dua_error().
score_explanations(Graph, Evidence, Explanations) ->
    ScoredExplanations = lists:map(fun(Explanation) ->
        Score = compute_explanation_score(Graph, Evidence, Explanation),
        {Explanation, Score}
    end, Explanations),
    {ok, ScoredExplanations}.

%% @private Compute score for an explanation.
-spec compute_explanation_score(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) -> float().
compute_explanation_score(Graph, Evidence, Explanation) ->
    try
        %% Create explanation evidence (assume all causes are active)
        ExplanationEvidence = maps:from_list([{Node, true} || Node <- Explanation]),
        %% Compute how well this explanation predicts the evidence
        case dua_forward:infer(Graph, ExplanationEvidence) of
            {ok, #{beliefs := Beliefs}} ->
                Likelihood = compute_evidence_likelihood(Beliefs, Evidence),
                %% Penalize complex explanations (Occam's razor)
                ComplexityPenalty = math:exp(-length(Explanation) * 0.1),
                Likelihood * ComplexityPenalty;
            _ ->
                0.0
        end
    catch
        _:_ ->
            0.0
    end.

%% @private Select best explanations.
-spec select_best_explanations([{[node_id()], float()}]) -> {ok, [[node_id()]]} | dua_error().
select_best_explanations(ScoredExplanations) ->
    %% Sort by score (descending) and take the best ones
    SortedExplanations = lists:sort(fun({_, S1}, {_, S2}) -> S1 >= S2 end, ScoredExplanations),
    BestExplanations = [Explanation || {Explanation, _Score} <- SortedExplanations],
    {ok, BestExplanations}.

%% @private Convert explanations to beliefs.
-spec explanations_to_beliefs(#dua_graph{}, [[node_id()]]) -> {ok, #{node_id() => belief()}} | dua_error().
explanations_to_beliefs(Graph, Explanations) ->
    %% Convert explanations to belief format
    case Explanations of
        [] ->
            {ok, #{}};
        [BestExplanation | _] ->
            %% Create beliefs for the best explanation
            ExplanationBeliefs = lists:foldl(fun(NodeId, BeliefAcc) ->
                case dua_graph:get_node(Graph, NodeId) of
                    {ok, #dua_node{states = States}} ->
                        %% Assume the cause is active
                        ActiveBelief = dua_belief:certain_belief(States, true),
                        maps:put(NodeId, ActiveBelief, BeliefAcc);
                    _ ->
                        BeliefAcc
                end
            end, #{}, BestExplanation),
            {ok, ExplanationBeliefs}
    end.

%% @private Extract explanation path.
-spec extract_explanation_path([[node_id()]]) -> [node_id()].
extract_explanation_path(Explanations) ->
    case Explanations of
        [] -> [];
        [BestExplanation | _] -> BestExplanation
    end.

%% @private Compute explanation confidence.
-spec compute_explanation_confidence([{[node_id()], float()}]) -> float().
compute_explanation_confidence(ScoredExplanations) ->
    case ScoredExplanations of
        [] -> 0.0;
        [{_, BestScore} | _] -> min(BestScore, 1.0)
    end.

%% @private Compute backward confidence.
-spec compute_backward_confidence(#{node_id() => belief()}) -> float().
compute_backward_confidence(CausalBeliefs) ->
    case map_size(CausalBeliefs) of
        0 -> 0.0;
        N ->
            %% Average confidence across all causal beliefs
            TotalConfidence = maps:fold(fun(_, Belief, Acc) ->
                case dua_belief:most_likely_state(Belief) of
                    {ok, {_State, Prob}} -> Acc + Prob;
                    _ -> Acc
                end
            end, 0.0, CausalBeliefs),
            TotalConfidence / N
    end.

%% @private Get explanation variables.
-spec get_explanation_variables(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, [node_id()]} | dua_error().
get_explanation_variables(Graph, Evidence) ->
    %% Get all non-evidence nodes as potential explanation variables
    {ok, AllNodes} = get_all_node_ids(Graph),
    EvidenceNodes = maps:keys(Evidence),
    Variables = AllNodes -- EvidenceNodes,
    {ok, Variables}.

%% @private Find optimal assignment for MPE.
-spec find_optimal_assignment(#dua_graph{}, #{node_id() => node_state()}, [node_id()]) ->
    {ok, #{node_id() => node_state()}} | dua_error().
find_optimal_assignment(Graph, Evidence, Variables) ->
    %% Simplified MPE - would implement dynamic programming or branch-and-bound
    %% For now, use greedy assignment based on marginal probabilities
    case dua_forward:infer(Graph, Evidence) of
        {ok, #{beliefs := Beliefs}} ->
            OptimalAssignment = maps:fold(fun(NodeId, Belief, AssignmentAcc) ->
                case lists:member(NodeId, Variables) of
                    true ->
                        case dua_belief:most_likely_state(Belief) of
                            {ok, {State, _Prob}} ->
                                maps:put(NodeId, State, AssignmentAcc);
                            _ ->
                                AssignmentAcc
                        end;
                    false ->
                        AssignmentAcc
                end
            end, Evidence, Beliefs),
            {ok, OptimalAssignment};
        Error ->
            Error
    end.

%% @private Compute assignment probability.
-spec compute_assignment_probability(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, float()} | dua_error().
compute_assignment_probability(Graph, Assignment) ->
    %% Compute P(assignment) using forward inference
    case dua_forward:infer(Graph, Assignment) of
        {ok, #{confidence := Confidence}} ->
            {ok, Confidence};
        _ ->
            {ok, 0.0}
    end.

%% @private Convert assignment to beliefs.
-spec assignment_to_beliefs(#dua_graph{}, #{node_id() => node_state()}) ->
    {ok, #{node_id() => belief()}} | dua_error().
assignment_to_beliefs(Graph, Assignment) ->
    %% Convert assignment to belief format
    {ok, AllNodes} = get_all_node_ids(Graph),
    Beliefs = lists:foldl(fun(NodeId, BeliefAcc) ->
        case maps:get(NodeId, Assignment, undefined) of
            undefined ->
                %% Use prior belief
                maps:put(NodeId, get_prior_belief(Graph, NodeId), BeliefAcc);
            AssignedState ->
                %% Create certain belief for assigned state
                case dua_graph:get_node(Graph, NodeId) of
                    {ok, #dua_node{states = States}} ->
                        CertainBelief = dua_belief:certain_belief(States, AssignedState),
                        maps:put(NodeId, CertainBelief, BeliefAcc);
                    _ ->
                        BeliefAcc
                end
        end
    end, #{}, AllNodes),
    {ok, Beliefs}.

%% @private Check if a node is an evidence node.
-spec is_evidence_node(node_id(), #{node_id() => node_state()}) -> boolean().
is_evidence_node(NodeId, Evidence) ->
    maps:is_key(NodeId, Evidence).

%% @private Get all node IDs in the graph.
-spec get_all_node_ids(#dua_graph{}) -> {ok, [node_id()]}.
get_all_node_ids(#dua_graph{nodes = Nodes}) ->
    {ok, maps:keys(Nodes)}.

