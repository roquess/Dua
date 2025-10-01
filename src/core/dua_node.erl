%%%-------------------------------------------------------------------
%%% @doc DUA Node Management Module
%%
%%% This module handles individual node operations, conditional probability
%%% tables (CPTs), and node-specific computations within the graph.
%%%-------------------------------------------------------------------
-module(dua_node).
-include("dua.hrl").

%% API
-export([
    %% Node creation and manipulation
    new/2,
    new/3,
    new_discrete/3,
    new_continuous/3,
    new_boolean/1,
    %% Node properties
    get_id/1,
    get_type/1,
    get_states/1,
    get_parents/1,
    get_children/1,
    get_belief/1,
    get_cpt/1,
    %% Node updates
    set_belief/2,
    set_cpt/2,
    add_parent/2,
    add_child/2,
    remove_parent/2,
    remove_child/2,
    %% CPT operations
    create_cpt/2,
    update_cpt/3,
    get_cpt_entry/2,
    set_cpt_entry/3,
    normalize_cpt/1,
    %% Node computations
    compute_likelihood/2,
    compute_prior/1,
    compute_posterior/2,
    sample_from_node/1,
    %% Validation
    validate_node/1,
    validate_cpt/2,
    is_consistent/1
]).

-export([create_deterministic_node/3, create_noisy_node/4, mutual_information/2, create_deterministic_cpt/3, create_noisy_cpt/4]).

%%%===================================================================
%%% Node Creation and Manipulation
%%%===================================================================

%% @doc Create a new node with ID and type
%% @param NodeId The unique identifier for the node.
%% @param Type The type of the node (e.g., boolean, discrete, continuous).
%% @return A new #dua_node{} record.
-spec new(node_id(), node_type()) -> #dua_node{}.
new(NodeId, Type) ->
    States = get_default_states(Type),
    new(NodeId, Type, States).

%% @doc Create a new node with ID, type, and states
%% @param NodeId The unique identifier for the node.
%% @param Type The type of the node.
%% @param States The possible states of the node.
%% @return A new #dua_node{} record.
-spec new(node_id(), node_type(), [node_state()]) -> #dua_node{}.
new(NodeId, Type, States) ->
    InitialBelief = dua_belief:uniform_belief(States),
    #dua_node{
        id = NodeId,
        type = Type,
        states = States,
        parents = [],
        children = [],
        cpt = #{},
        current_belief = InitialBelief
    }.

%% @doc Create a discrete node with specific states
%% @param NodeId The unique identifier for the node.
%% @param States The possible states of the node.
%% @param InitialBelief The initial belief distribution.
%% @return A new #dua_node{} record.
-spec new_discrete(node_id(), [node_state()], belief()) -> #dua_node{}.
new_discrete(NodeId, States, InitialBelief) ->
    #dua_node{
        id = NodeId,
        type = discrete,
        states = States,
        parents = [],
        children = [],
        cpt = #{},
        current_belief = InitialBelief
    }.

%% @doc Create a continuous node with range parameters
%% @param NodeId The unique identifier for the node.
%% @param {Min, Max} The range of the continuous variable.
%% @param Parameters Additional parameters (e.g., number of bins).
%% @return A new #dua_node{} record.
-spec new_continuous(node_id(), {float(), float()}, map()) -> #dua_node{}.
new_continuous(NodeId, {Min, Max}, Parameters) ->
    %% For continuous nodes, we discretize the range
    NumBins = maps:get(num_bins, Parameters, 10),
    States = discretize_range(Min, Max, NumBins),
    InitialBelief = dua_belief:uniform_belief(States),
    #dua_node{
        id = NodeId,
        type = continuous,
        states = States,
        parents = [],
        children = [],
        cpt = #{range => {Min, Max}, bins => NumBins},
        current_belief = InitialBelief
    }.

%% @doc Create a boolean node
%% @param NodeId The unique identifier for the node.
%% @return A new #dua_node{} record.
-spec new_boolean(node_id()) -> #dua_node{}.
new_boolean(NodeId) ->
    new(NodeId, boolean, [true, false]).

%%%===================================================================
%%% Node Properties
%%%===================================================================

%% @doc Get node ID
%% @return The node's ID.
-spec get_id(#dua_node{}) -> node_id().
get_id(#dua_node{id = Id}) -> Id.

%% @doc Get node type
%% @return The node's type.
-spec get_type(#dua_node{}) -> node_type().
get_type(#dua_node{type = Type}) -> Type.

%% @doc Get node states
%% @return The node's possible states.
-spec get_states(#dua_node{}) -> [node_state()].
get_states(#dua_node{states = States}) -> States.

%% @doc Get node parents
%% @return The node's parent IDs.
-spec get_parents(#dua_node{}) -> [node_id()].
get_parents(#dua_node{parents = Parents}) -> Parents.

%% @doc Get node children
%% @return The node's child IDs.
-spec get_children(#dua_node{}) -> [node_id()].
get_children(#dua_node{children = Children}) -> Children.

%% @doc Get current belief
%% @return The node's current belief distribution.
-spec get_belief(#dua_node{}) -> belief().
get_belief(#dua_node{current_belief = Belief}) -> Belief.

%% @doc Get conditional probability table
%% @return The node's CPT.
-spec get_cpt(#dua_node{}) -> map().
get_cpt(#dua_node{cpt = CPT}) -> CPT.

%%%===================================================================
%%% Node Updates
%%%===================================================================

%% @doc Set node belief
%% @param NewBelief The new belief distribution.
%% @return {ok, #dua_node{}} if successful, otherwise dua_error().
-spec set_belief(#dua_node{}, belief()) -> {ok, #dua_node{}} | dua_error().
set_belief(Node, NewBelief) ->
    case dua_belief:is_valid_belief(NewBelief) of
        true ->
            {ok, Node#dua_node{current_belief = NewBelief}};
        false ->
            {error, {invalid_belief, NewBelief}}
    end.

%% @doc Set conditional probability table
%% @param NewCPT The new CPT.
%% @return {ok, #dua_node{}} if successful, otherwise dua_error().
-spec set_cpt(#dua_node{}, map()) -> {ok, #dua_node{}} | dua_error().
set_cpt(Node, NewCPT) ->
    case validate_cpt(Node, NewCPT) of
        ok ->
            {ok, Node#dua_node{cpt = NewCPT}};
        Error ->
            Error
    end.

%% @doc Add a parent to the node
%% @param ParentId The ID of the parent node.
%% @return The updated node.
-spec add_parent(#dua_node{}, node_id()) -> #dua_node{}.
add_parent(#dua_node{parents = Parents} = Node, ParentId) ->
    case lists:member(ParentId, Parents) of
        true ->
            Node;  % Parent already exists
        false ->
            Node#dua_node{parents = [ParentId | Parents]}
    end.

%% @doc Add a child to the node
%% @param ChildId The ID of the child node.
%% @return The updated node.
-spec add_child(#dua_node{}, node_id()) -> #dua_node{}.
add_child(#dua_node{children = Children} = Node, ChildId) ->
    case lists:member(ChildId, Children) of
        true ->
            Node;  % Child already exists
        false ->
            Node#dua_node{children = [ChildId | Children]}
    end.

%% @doc Remove a parent from the node
%% @param ParentId The ID of the parent node.
%% @return The updated node.
-spec remove_parent(#dua_node{}, node_id()) -> #dua_node{}.
remove_parent(#dua_node{parents = Parents} = Node, ParentId) ->
    UpdatedParents = lists:delete(ParentId, Parents),
    Node#dua_node{parents = UpdatedParents}.

%% @doc Remove a child from the node
%% @param ChildId The ID of the child node.
%% @return The updated node.
-spec remove_child(#dua_node{}, node_id()) -> #dua_node{}.
remove_child(#dua_node{children = Children} = Node, ChildId) ->
    UpdatedChildren = lists:delete(ChildId, Children),
    Node#dua_node{children = UpdatedChildren}.

%%%===================================================================
%%% CPT Operations
%%%===================================================================

%% @doc Create a CPT for a node based on its parents
%% @param ParentStates Map of parent node IDs to their possible states.
%% @return The generated CPT.
-spec create_cpt(#dua_node{}, #{node_id() => [node_state()]}) -> map().

create_cpt(#dua_node{states = States, parents = Parents}, ParentStates) ->
    case Parents of
        [] ->
            %% No parents - just prior probabilities
            dua_belief:uniform_belief(States);
        _ ->
            %% Generate all parent combinations
            ParentCombinations = generate_parent_combinations(Parents, ParentStates),
            %% Create CPT entry for each combination
            maps:from_list([{Combination, dua_belief:uniform_belief(States)}
                           || Combination <- ParentCombinations])
    end.

%% @doc Update a specific entry in the CPT
%% @param ParentAssignment The parent state combination.
%% @param NewBelief The new belief for this combination.
%% @return {ok, #dua_node{}} if successful, otherwise dua_error().
-spec update_cpt(#dua_node{}, map(), belief()) -> {ok, #dua_node{}} | dua_error().
update_cpt(#dua_node{cpt = CPT} = Node, ParentAssignment, NewBelief) ->
    case dua_belief:is_valid_belief(NewBelief) of
        true ->
            UpdatedCPT = maps:put(ParentAssignment, NewBelief, CPT),
            {ok, Node#dua_node{cpt = UpdatedCPT}};
        false ->
            {error, {invalid_belief, NewBelief}}
    end.

%% @doc Get a specific entry from the CPT
%% @param ParentAssignment The parent state combination.
%% @return The belief for this combination, or undefined.
-spec get_cpt_entry(#dua_node{}, map()) -> belief() | undefined.
get_cpt_entry(#dua_node{cpt = CPT}, ParentAssignment) ->
    maps:get(ParentAssignment, CPT, undefined).

%% @doc Set a specific entry in the CPT
%% @param ParentAssignment The parent state combination.
%% @param Belief The new belief for this combination.
%% @return {ok, #dua_node{}} if successful, otherwise dua_error().
-spec set_cpt_entry(#dua_node{}, map(), belief()) -> {ok, #dua_node{}} | dua_error().
set_cpt_entry(Node, ParentAssignment, Belief) ->
    update_cpt(Node, ParentAssignment, Belief).

%% @doc Normalize all entries in the CPT
%% @return The updated node.
-spec normalize_cpt(#dua_node{}) -> #dua_node{}.
normalize_cpt(#dua_node{cpt = CPT} = Node) ->
    NormalizedCPT = maps:map(fun(_Key, Belief) ->
        dua_belief:normalize(Belief)
    end, CPT),
    Node#dua_node{cpt = NormalizedCPT}.

%%%===================================================================
%%% Node Computations
%%%===================================================================

%% @doc Compute likelihood of evidence given node state
%% @param EvidenceState The evidence state.
%% @return The likelihood as a probability.
-spec compute_likelihood(#dua_node{}, node_state()) -> probability().
compute_likelihood(#dua_node{current_belief = Belief}, EvidenceState) ->
    maps:get(EvidenceState, Belief, 0.0).

%% @doc Compute prior probability distribution
%% @return The prior belief.
-spec compute_prior(#dua_node{}) -> belief().
compute_prior(#dua_node{parents = [], current_belief = Belief}) ->
    %% For root nodes, current belief is the prior
    Belief;
compute_prior(#dua_node{states = States}) ->
    %% For non-root nodes, use uniform prior
    dua_belief:uniform_belief(States).

%% @doc Compute posterior given evidence
%% @param Evidence The evidence to apply.
%% @return The posterior belief.
-spec compute_posterior(#dua_node{}, #{node_id() => node_state()}) -> belief().
compute_posterior(#dua_node{current_belief = Prior} = Node, Evidence) ->
    %% Apply Bayes' rule
    maps:fold(fun(EvidenceVar, EvidenceState, BeliefAcc) ->
        case EvidenceVar =:= Node#dua_node.id of
            true ->
                %% Direct evidence on this node
                dua_belief:apply_evidence(BeliefAcc, #{EvidenceState => 1.0});
            false ->
                %% Indirect evidence - would need to propagate through CPT
                BeliefAcc
        end
    end, Prior, Evidence).

%% @doc Sample a state from the node's current belief
%% @return A sampled state.
-spec sample_from_node(#dua_node{}) -> node_state().
sample_from_node(#dua_node{current_belief = Belief}) ->
    dua_probability:sample_from_distribution(Belief).

%%%===================================================================
%%% Validation
%%%===================================================================

%% @doc Validate node consistency
%% @return ok if valid, otherwise dua_error().
-spec validate_node(#dua_node{}) -> ok | dua_error().
validate_node(#dua_node{states = States, current_belief = Belief, cpt = CPT} = Node) ->
    %% Check states are not empty
    case States of
        [] ->
            {error, empty_states};
        _ ->
            %% Check belief is valid
            case dua_belief:is_valid_belief(Belief) of
                false ->
                    {error, invalid_belief};
                true ->
                    %% Check belief states match node states
                    BeliefStates = sets:from_list(maps:keys(Belief)),
                    NodeStates = sets:from_list(States),
                    case sets:is_subset(BeliefStates, NodeStates) of
                        false ->
                            {error, belief_state_mismatch};
                        true ->
                            %% Validate CPT
                            validate_cpt(Node, CPT)
                    end
            end
    end.

%% @doc Validate conditional probability table
%% @return ok if valid, otherwise dua_error().
-spec validate_cpt(#dua_node{}, map()) -> ok | dua_error().
validate_cpt(#dua_node{states = States, parents = Parents}, CPT) ->
    case Parents of
        [] ->
            %% Root node - CPT should be empty or contain only prior
            case map_size(CPT) of
                0 -> ok;
                1 ->
                    %% Check if it contains a valid belief
                    case maps:values(CPT) of
                        [Belief] ->
                            case dua_belief:is_valid_belief(Belief) of
                                true -> ok;
                                false -> {error, invalid_cpt_belief}
                            end;
                        _ -> {error, invalid_root_cpt}
                    end;
                _ -> {error, invalid_root_cpt}
            end;
        _ ->
            %% Non-root node - validate all CPT entries
            maps:fold(fun(ParentAssignment, Belief, ok) ->
                %% Check if belief is valid
                case dua_belief:is_valid_belief(Belief) of
                    false ->
                        {error, {invalid_cpt_belief, ParentAssignment}};
                    true ->
                        %% Check if belief states match node states
                        BeliefStates = sets:from_list(maps:keys(Belief)),
                        NodeStates = sets:from_list(States),
                        case sets:is_subset(BeliefStates, NodeStates) of
                            false ->
                                {error, {cpt_state_mismatch, ParentAssignment}};
                            true ->
                                ok
                        end
                end;
            (_, _, Error) -> Error
            end, ok, CPT)
    end.

%% @doc Check if node is consistent with its relationships
%% @return boolean().
-spec is_consistent(#dua_node{}) -> boolean().
is_consistent(Node) ->
    case validate_node(Node) of
        ok -> true;
        _ -> false
    end.

%%%===================================================================
%%% Internal Functions
%%%===================================================================

%% @private Get default states for a node type
-spec get_default_states(node_type()) -> [node_state()].
get_default_states(boolean) -> [true, false];
get_default_states(discrete) -> [low, medium, high];
get_default_states(continuous) -> discretize_range(0.0, 1.0, 10);
get_default_states(_) -> [unknown].

%% @private Discretize a continuous range into bins
-spec discretize_range(float(), float(), pos_integer()) -> [float()].
discretize_range(Min, Max, NumBins) ->
    StepSize = (Max - Min) / NumBins,
    [Min + I * StepSize || I <- lists:seq(0, NumBins - 1)].

%% @private Generate all possible parent state combinations
-spec generate_parent_combinations([node_id()], #{node_id() => [node_state()]}) -> [map()].
generate_parent_combinations([], _ParentStates) ->
    [{}];
generate_parent_combinations([Parent | RestParents], ParentStates) ->
    States = maps:get(Parent, ParentStates, [unknown]),
    RestCombinations = generate_parent_combinations(RestParents, ParentStates),
    lists:flatten([
        [maps:put(Parent, State, Combination) || State <- States]
        || Combination <- RestCombinations
    ]).

%%%===================================================================
%%% Advanced Node Operations (Stub/Unused)
%%%===================================================================

%% @doc (STUB) Create a deterministic node (always produces same output for same input)
%% @param Function The deterministic function.
%% @param Parents The parent node IDs.
%% @return A new #dua_node{} record.
%% @note This function is currently a stub and not used.
-spec create_deterministic_node(node_id(), [node_id()], fun()) -> #dua_node{}.
create_deterministic_node(NodeId, _Parents, _Function) ->
    States = [true, false],  % Simplified for boolean output
    Node = new(NodeId, discrete, States),
    %% TODO: Implement deterministic CPT logic
    Node.

%% @doc (STUB) Create a noisy node (adds noise to deterministic function)
%% @param Function The deterministic function.
%% @param Parents The parent node IDs.
%% @param NoiseLevel The level of noise to add.
%% @return A new #dua_node{} record.
%% @note This function is currently a stub and not used.
-spec create_noisy_node(node_id(), [node_id()], fun(), float()) -> #dua_node{}.
create_noisy_node(NodeId, _Parents, _Function, _NoiseLevel) ->
    States = [true, false],
    Node = new(NodeId, discrete, States),
    %% TODO: Implement noisy CPT logic
    Node.

%% @doc (STUB) Compute mutual information between node and evidence
%% @param Evidence The evidence to compare with.
%% @return The mutual information value.
%% @note This function is currently a stub and not used.
-spec mutual_information(#dua_node{}, #{node_id() => node_state()}) -> float().
mutual_information(#dua_node{current_belief = NodeBelief}, _Evidence) ->
    %% Simplified MI calculation
    %% TODO: Implement full mutual information logic
    NodeEntropy = dua_belief:entropy(NodeBelief),
    %% Approximate conditional entropy
    ConditionalEntropy = NodeEntropy * 0.5,  % Simplified
    NodeEntropy - ConditionalEntropy.

%% @private (STUB) Create a deterministic CPT
%% @note This function is currently a stub and not used.
-spec create_deterministic_cpt(fun(), [node_id()], [node_state()]) -> map().
create_deterministic_cpt(_Function, _Parents, States) ->
    %% TODO: Implement deterministic CPT logic
    #{#{} => dua_belief:uniform_belief(States)}.

%% @private (STUB) Create a noisy CPT
%% @note This function is currently a stub and not used.
-spec create_noisy_cpt(fun(), [node_id()], [node_state()], float()) -> map().
create_noisy_cpt(_Function, _Parents, States, _NoiseLevel) ->
    %% TODO: Implement noisy CPT logic
    #{#{} => dua_belief:uniform_belief(States)}.

