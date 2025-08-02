%%%-------------------------------------------------------------------
%%% @doc DUA Belief Management Module
%%% 
%%% This module handles belief representation, updates, and operations
%%% including belief combination, normalization, and evidence integration.
%%%-------------------------------------------------------------------

-module(dua_belief).

-include("dua.hrl").

-export([
    %% Belief creation and manipulation
    new_belief/1,
    new_belief/2,
    uniform_belief/1,
    certain_belief/2,
    
    %% Belief operations
    normalize/1,
    combine/2,
    combine/3,
    marginalize/2,
    condition/3,
    
    %% Evidence handling
    apply_evidence/2,
    evidence_strength/2,
    conflict_measure/2,
    
    %% Belief queries
    most_likely_state/1,
    entropy/1,
    kl_divergence/2,
    
    %% Utility functions
    is_valid_belief/1,
    belief_distance/2,
    interpolate/3
]).

%%%===================================================================
%%% Belief Creation and Manipulation
%%%===================================================================

%% @doc Create a new belief from a list of states with equal probability
-spec new_belief([node_state()]) -> belief().
new_belief(States) ->
    uniform_belief(States).

%% @doc Create a new belief from states and probabilities
-spec new_belief([node_state()], [probability()]) -> belief() | dua_error().
new_belief(States, Probabilities) when length(States) =:= length(Probabilities) ->
    case lists:all(fun(P) -> P >= 0.0 andalso P =< 1.0 end, Probabilities) of
        false ->
            {error, invalid_probabilities};
        true ->
            Belief = maps:from_list(lists:zip(States, Probabilities)),
            case abs(lists:sum(Probabilities) - 1.0) < ?DEFAULT_PRECISION of
                true -> Belief;
                false -> normalize(Belief)
            end
    end;
new_belief(States, Probabilities) ->
    {error, {length_mismatch, length(States), length(Probabilities)}}.

%% @doc Create a uniform belief over given states
-spec uniform_belief([node_state()]) -> belief().
uniform_belief([]) ->
    #{};
uniform_belief(States) ->
    Probability = 1.0 / length(States),
    maps:from_list([{State, Probability} || State <- States]).

%% @doc Create a certain belief (probability 1.0 for one state)
-spec certain_belief([node_state()], node_state()) -> belief() | dua_error().
certain_belief(States, CertainState) ->
    case lists:member(CertainState, States) of
        false ->
            {error, {invalid_state, CertainState}};
        true ->
            maps:from_list([
                {State, if State =:= CertainState -> 1.0; true -> 0.0 end}
                || State <- States
            ])
    end.

%%%===================================================================
%%% Belief Operations
%%%===================================================================

%% @doc Normalize a belief to ensure probabilities sum to 1.0
-spec normalize(belief()) -> belief().
normalize(Belief) when map_size(Belief) =:= 0 ->
    Belief;
normalize(Belief) ->
    Total = maps:fold(fun(_, Prob, Sum) -> Sum + Prob end, 0.0, Belief),
    case Total > ?DEFAULT_PRECISION of
        true ->
            maps:map(fun(_, Prob) -> Prob / Total end, Belief);
        false ->
            %% If all probabilities are essentially zero, create uniform
            States = maps:keys(Belief),
            uniform_belief(States)
    end.

%% @doc Combine two beliefs using multiplication (independence assumption)
-spec combine(belief(), belief()) -> belief().
combine(Belief1, Belief2) ->
    combine(Belief1, Belief2, multiply).

%% @doc Combine two beliefs using specified method
-spec combine(belief(), belief(), multiply | average | max | min) -> belief().
combine(Belief1, Belief2, Method) ->
    CommonStates = sets:to_list(
        sets:intersection(
            sets:from_list(maps:keys(Belief1)),
            sets:from_list(maps:keys(Belief2))
        )
    ),
    
    Combined = maps:from_list([
        {State, combine_probabilities(
            maps:get(State, Belief1, 0.0),
            maps:get(State, Belief2, 0.0),
            Method
        )}
        || State <- CommonStates
    ]),
    
    normalize(Combined).

%% @doc Marginalize a belief by removing certain states
-spec marginalize(belief(), [node_state()]) -> belief().
marginalize(Belief, StatesToRemove) ->
    FilteredBelief = maps:without(StatesToRemove, Belief),
    normalize(FilteredBelief).

%% @doc Condition a belief on evidence
-spec condition(belief(), node_state(), probability()) -> belief().
condition(Belief, EvidenceState, EvidenceStrength) ->
    ConditionedBelief = maps:map(fun(State, Prob) ->
        if State =:= EvidenceState ->
            Prob * EvidenceStrength + (1.0 - EvidenceStrength) * Prob;
        true ->
            Prob * (1.0 - EvidenceStrength) + EvidenceStrength * 0.0
        end
    end, Belief),
    normalize(ConditionedBelief).

%%%===================================================================
%%% Evidence Handling
%%%===================================================================

%% @doc Apply evidence to a belief
-spec apply_evidence(belief(), #{node_state() => probability()}) -> belief().
apply_evidence(Belief, Evidence) ->
    UpdatedBelief = maps:fold(fun(EvidenceState, EvidenceProb, BeliefAcc) ->
        case maps:is_key(EvidenceState, BeliefAcc) of
            true ->
                maps:put(EvidenceState, EvidenceProb, BeliefAcc);
            false ->
                BeliefAcc
        end
    end, Belief, Evidence),
    normalize(UpdatedBelief).

%% @doc Calculate the strength of evidence for a particular state
-spec evidence_strength(belief(), node_state()) -> probability().
evidence_strength(Belief, State) ->
    maps:get(State, Belief, 0.0).

%% @doc Measure conflict between two beliefs
-spec conflict_measure(belief(), belief()) -> probability().
conflict_measure(Belief1, Belief2) ->
    CommonStates = sets:to_list(
        sets:intersection(
            sets:from_list(maps:keys(Belief1)),
            sets:from_list(maps:keys(Belief2))
        )
    ),
    
    Conflict = lists:foldl(fun(State, Acc) ->
        P1 = maps:get(State, Belief1, 0.0),
        P2 = maps:get(State, Belief2, 0.0),
        Acc + abs(P1 - P2)
    end, 0.0, CommonStates),
    
    case length(CommonStates) of
        0 -> 1.0;  % Complete conflict if no common states
        N -> Conflict / N
    end.

%%%===================================================================
%%% Belief Queries
%%%===================================================================

%% @doc Find the most likely state in a belief
-spec most_likely_state(belief()) -> {ok, {node_state(), probability()}} | {error, empty_belief}.
most_likely_state(Belief) when map_size(Belief) =:= 0 ->
    {error, empty_belief};
most_likely_state(Belief) ->
    {State, Prob} = maps:fold(fun(S, P, {MaxS, MaxP}) ->
        if P > MaxP -> {S, P};
           true -> {MaxS, MaxP}
        end
    end, {undefined, -1.0}, Belief),
    {ok, {State, Prob}}.

%% @doc Calculate the entropy of a belief
-spec entropy(belief()) -> float().
entropy(Belief) ->
    maps:fold(fun(_, Prob, Acc) ->
        if Prob > 0.0 ->
            Acc - Prob * math:log2(Prob);
        true ->
            Acc
        end
    end, 0.0, Belief).

%% @doc Calculate KL divergence between two beliefs
-spec kl_divergence(belief(), belief()) -> float() | {error, term()}.
kl_divergence(Belief1, Belief2) ->
    CommonStates = sets:to_list(
        sets:intersection(
            sets:from_list(maps:keys(Belief1)),
            sets:from_list(maps:keys(Belief2))
        )
    ),
    
    case CommonStates of
        [] ->
            {error, no_common_states};
        _ ->
            KL = lists:foldl(fun(State, Acc) ->
                P1 = maps:get(State, Belief1, 0.0),
                P2 = maps:get(State, Belief2, 0.0),
                if P1 > 0.0 andalso P2 > 0.0 ->
                    Acc + P1 * math:log2(P1 / P2);
                true ->
                    Acc
                end
            end, 0.0, CommonStates),
            KL
    end.

%%%===================================================================
%%% Utility Functions
%%%===================================================================

%% @doc Check if a belief is valid (probabilities sum to ~1.0)
-spec is_valid_belief(belief()) -> boolean().
is_valid_belief(Belief) ->
    Total = maps:fold(fun(_, Prob, Sum) -> Sum + Prob end, 0.0, Belief),
    AllValid = maps:fold(fun(_, Prob, Valid) -> 
        Valid andalso Prob >= 0.0 andalso Prob =< 1.0 
    end, true, Belief),
    AllValid andalso abs(Total - 1.0) < ?DEFAULT_PRECISION.

%% @doc Calculate distance between two beliefs
-spec belief_distance(belief(), belief()) -> float().
belief_distance(Belief1, Belief2) ->
    AllStates = sets:to_list(
        sets:union(
            sets:from_list(maps:keys(Belief1)),
            sets:from_list(maps:keys(Belief2))
        )
    ),
    
    Distance = lists:foldl(fun(State, Acc) ->
        P1 = maps:get(State, Belief1, 0.0),
        P2 = maps:get(State, Belief2, 0.0),
        Acc + (P1 - P2) * (P1 - P2)
    end, 0.0, AllStates),
    
    math:sqrt(Distance).

%% @doc Interpolate between two beliefs
-spec interpolate(belief(), belief(), float()) -> belief().
interpolate(Belief1, Belief2, Alpha) when Alpha >= 0.0, Alpha =< 1.0 ->
    AllStates = sets:to_list(
        sets:union(
            sets:from_list(maps:keys(Belief1)),
            sets:from_list(maps:keys(Belief2))
        )
    ),
    
    Interpolated = maps:from_list([
        {State, Alpha * maps:get(State, Belief1, 0.0) + 
                (1.0 - Alpha) * maps:get(State, Belief2, 0.0)}
        || State <- AllStates
    ]),
    
    normalize(Interpolated).

%%%===================================================================
%%% Internal Functions
%%%===================================================================

%% @private
combine_probabilities(P1, P2, multiply) ->
    P1 * P2;
combine_probabilities(P1, P2, average) ->
    (P1 + P2) / 2.0;
combine_probabilities(P1, P2, max) ->
    max(P1, P2);
combine_probabilities(P1, P2, min) ->
    min(P1, P2).
