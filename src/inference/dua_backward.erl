%%%-------------------------------------------------------------------
%%% @doc DUA Backward Inference Engine
%%% 
%%% This module implements backward inference (effect → cause) using
%%% Bayesian inversion, abductive reasoning, and causal analysis.
%%%-------------------------------------------------------------------

-module(dua_backward).

-include("dua.hrl").

-export([
    %% Main inference functions
    infer/2,
    infer/3,
    
    %% Specific inference algorithms
    bayesian_inversion/3,
    abductive_reasoning/3,
    most_probable_explanation/3,
    
    %% Utility functions
    explain/3,
    find_causes/3,
    causal_strength/4
]).

%%%===================================================================
%%% Main Inference Functions
%%%===================================================================

%% @doc Perform backward inference with evidence
-spec infer(#dua_graph{}, evidence()) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence) ->
    infer(Graph, Evidence, #{algorithm => bayesian_inversion}).

%% @doc Perform backward inference with options
-spec infer(#dua_graph{}, evidence(), map()) -> {ok, query_result()} | dua_error().
infer(Graph, Evidence, Options) ->
    Algorithm = maps:get(algorithm, Options, bayesian_inversion),
    
    case validate_evidence(Graph, Evidence) of
        ok ->
            execute_backward_inference(Graph, Evidence, Algorithm, Options);
        Error ->
            Error
    end.

%%%===================================================================
%%% Inference Algorithms
%%%===================================================================

%% @doc Bayesian inversion for backward inference
-spec bayesian_inversion(#dua_graph{}, evidence(), map()) -> 
    {ok, query_result()} | dua_error().
bayesian_inversion(Graph, Evidence, Options) ->
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

%% @doc Abductive reasoning for finding best explanations
-spec abductive_reasoning(#dua_graph{}, evidence(), map()) -> 
    {ok, query_result()} | dua_error().
abductive_reasoning(Graph, Evidence, Options) ->
    MaxExplanations = maps:get(max_explanations, Options, 10),
    
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

%% @doc Most Probable Explanation (MPE) algorithm
-spec most_probable_explanation(#dua_graph{}, evidence(), map()) -> 
    {ok, query_result()} | dua_error().
most_probable_explanation(Graph, Evidence, Options) ->
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

%% @doc Explain observed effects by finding likely causes
-spec explain(#dua_graph{}, evidence(), map()) -> 
    {ok, #{explanation => term(), confidence => float()}} | dua_error().
explain(Graph, Evidence, Options) ->
    case infer(Graph, Evidence, Options#{algorithm => abductive_reasoning}) of
        {ok, #{explanations := Explanations, confidence := Confidence}} ->
            BestExplanation = case Explanations of
                [Best | _] -> Best;
                [] -> no_explanation_found
            end,
            {ok, #{explanation => BestExplanation, confidence => Confidence}};
        Error ->
            Error
    end.

%% @doc Find potential causes for given effects
-spec find_causes(#dua_graph{}, evidence(), map()) -> 
    {ok, [node_id()]} | dua_error().
find_causes(Graph, Evidence, Options) ->
    Threshold = maps:get(threshold, Options, 0.1),
    
    case infer(Graph, Evidence) of
        {ok, #{beliefs := Beliefs}} ->
            Causes = maps:fold(fun(NodeId, Belief, CauseAcc) ->
                case is_evidence_node(NodeId, Evidence) of
                    true -> CauseAcc;  % Skip evidence nodes
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

%% @doc Calculate causal strength between cause and effect
-spec causal_strength(#dua_graph{}, node_id(), node_id(), evidence()) -> 
    {ok, float()} | dua_error().
causal_strength(Graph, CauseNode, EffectNode, BaseEvidence) ->
    try
        %% Compute P(effect|cause=true, evidence)
        CauseTrue = maps:put(CauseNode, true, BaseEvidence),
        {ok, #{beliefs := BeliefsTrue}} = dua_forward:infer(Graph, CauseTrue),
        
        %% Compute P(effect|cause=false, evidence)
        CauseFalse = maps:put(CauseNode, false, BaseEvidence),
        {ok, #{beliefs := BeliefsFalse}} = dua_forward:infer(Graph, CauseFalse),
        
        %% Calculate causal strength as difference in probabilities
        case {maps:get(EffectNode, BeliefsTrue, #{}), 
              maps:get(EffectNode, BeliefsFalse, #{})} of
            {BeliefTrue, BeliefFalse} ->
                {ok, {_StateTrue, ProbTrue}} = dua_belief:most_likely_state(BeliefTrue),
                {ok, {_StateFalse, ProbFalse}} = dua_belief:most_likely_state(BeliefFalse),
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
execute_backward_inference(Graph, Evidence, Algorithm, Options) ->
    case Algorithm of
        bayesian_inversion ->
            bayesian_inversion(Graph, Evidence, Options);
        abductive_reasoning ->
            abductive_reasoning(Graph, Evidence, Options);
        most_probable_explanation ->
            most_probable_explanation(Graph, Evidence, Options);
        _ ->
            {error, {unknown_algorithm, Algorithm}}
    end.

%% @private
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

%% @private
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

%% @private
compute_posterior_belief(Graph, CauseNode, Evidence) ->
    try
        %% Use forward inference to compute P(evidence|cause) for each cause state
        case dua_graph:get_node(Graph, CauseNode) of
            {ok, #dua_node{states = States}} ->
                %% Compute likelihood for each state of the cause
                Likelihoods = lists:foldl(fun(State, LikelihoodAcc) ->
                    CauseEvidence = maps:put(CauseNode, State, #{}),
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

%% @private
get_prior_belief(Graph, NodeId) ->
    case dua_graph:get_node(Graph, NodeId) of
        {ok, #dua_node{current_belief = Belief}} ->
            Belief;
        {ok, #dua_node{states = States}} ->
            dua_belief:uniform_belief(States);
        _ ->
            #{}
    end.

%% @private
compute_evidence_likelihood(Beliefs, Evidence) ->
    %% Compute how well the beliefs match the evidence
    maps:fold(fun(EvidenceNode, EvidenceState, LikelihoodAcc) ->
        case maps:get(EvidenceNode, Beliefs, #{}) of
            NodeBelief ->
                StateProb = maps:get(EvidenceState, NodeBelief, 0.0),
                LikelihoodAcc * StateProb;
            _ ->
                0.0
        end
    end, 1.0, Evidence).

%% @private
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

%% @private
generate_explanations(Graph, Evidence, MaxExplanations) ->
    %% Generate possible explanations by exploring combinations of causes
    {ok, PotentialCauses} = identify_potential_causes(Graph, Evidence),
    
    %% Generate combinations of causes (simplified - could use more sophisticated search)
    Explanations = generate_cause_combinations(PotentialCauses, MaxExplanations),
    
    {ok, Explanations}.

%% @private
generate_cause_combinations(Causes, MaxCombinations) ->
    %% Generate all possible combinations up to MaxCombinations
    AllCombinations = generate_all_combinations(Causes),
    lists:sublist(AllCombinations, MaxCombinations).

%% @private
generate_all_combinations([]) ->
    [[]];
generate_all_combinations([H | T]) ->
    RestCombinations = generate_all_combinations(T),
    WithH = [[H | Combo] || Combo <- RestCombinations],
    WithoutH = RestCombinations,
    WithH ++ WithoutH.

%% @private
score_explanations(Graph, Evidence, Explanations) ->
    ScoredExplanations = lists:map(fun(Explanation) ->
        Score = compute_explanation_score(Graph, Evidence, Explanation),
        {Explanation, Score}
    end, Explanations),
    
    {ok, ScoredExplanations}.

%% @private
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

%% @private
select_best_explanations(ScoredExplanations) ->
    %% Sort by score (descending) and take the best ones
    SortedExplanations = lists:sort(fun({_, S1}, {_, S2}) -> S1 >= S2 end, ScoredExplanations),
    BestExplanations = [Explanation || {Explanation, _Score} <- SortedExplanations],
    
    {ok, BestExplanations}.

%% @private
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

%% @private
extract_explanation_path(Explanations) ->
    case Explanations of
        [] -> [];
        [BestExplanation | _] -> BestExplanation
    end.

%% @private
compute_explanation_confidence(ScoredExplanations) ->
    case ScoredExplanations of
        [] -> 0.0;
        [{_, BestScore} | _] -> min(BestScore, 1.0)
    end.

%% @private
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

%% @private
get_explanation_variables(Graph, Evidence) ->
    %% Get all non-evidence nodes as potential explanation variables
    {ok, AllNodes} = get_all_node_ids(Graph),
    EvidenceNodes = maps:keys(Evidence),
    Variables = AllNodes -- EvidenceNodes,
    {ok, Variables}.

%% @private
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

%% @private
compute_assignment_probability(Graph, Assignment) ->
    %% Compute P(assignment) using forward inference
    case dua_forward:infer(Graph, Assignment) of
        {ok, #{confidence := Confidence}} ->
            {ok, Confidence};
        _ ->
            {ok, 0.0}
    end.

%% @private
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

%% @private
is_evidence_node(NodeId, Evidence) ->
    maps:is_key(NodeId, Evidence).

%% @private
get_all_node_ids(#dua_graph{nodes = Nodes}) ->
    {ok, maps:keys(Nodes)}.
