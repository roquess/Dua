%%%-------------------------------------------------------------------
%%% @doc DUA - Main header file
%%% @author Steve Roques
%%%-------------------------------------------------------------------

-ifndef(DUA_HRL).
-define(DUA_HRL, true).

-define(DUA_VERSION, "0.1.0").

%% Default values
-define(DEFAULT_PRECISION, 0.001).
-define(MAX_ITERATIONS, 1000).
-define(MIN_PROBABILITY, 0.0001).
-define(MAX_PROBABILITY, 0.9999).

%% Node types
-type node_id() :: atom() | binary().
-type node_type() :: discrete | continuous | boolean.
-type node_state() :: term().

%% Probability types
-type probability() :: float().  % 0.0 to 1.0
-type belief() :: #{node_state() => probability()}.
-type evidence() :: #{node_id() => node_state()}.

%% Graph structures
-record(dua_node, {
    id :: node_id(),
    type :: node_type(),
    states :: [node_state()],
    parents :: [node_id()],
    children :: [node_id()],
    cpt :: map(),  % Conditional Probability Table
    current_belief :: belief()
}).

-record(dua_edge, {
    from :: node_id(),
    to :: node_id(),
    weight :: probability()
}).

-record(dua_graph, {
    nodes :: #{node_id() => #dua_node{}},
    edges :: [#dua_edge{}],
    topology :: directed_acyclic_graph,
    metadata :: map()
}).

%% Query types
-type query_type() :: forward | backward | bidirectional.
-type query_result() :: #{
    node_id() => belief(),
    inference_path => [node_id()],
    confidence => probability(),
    iterations => non_neg_integer()
}.

%% API types
-record(dua_config, {
    precision = ?DEFAULT_PRECISION :: float(),
    max_iterations = ?MAX_ITERATIONS :: non_neg_integer(),
    debug = false :: boolean()
}).

%% Error types
-type dua_error() :: {error, {atom(), term()}}.

%% Inference direction
-type inference_direction() :: forward | backward.

%% Export common types
-export_type([
    node_id/0,
    node_type/0,
    node_state/0,
    probability/0,
    belief/0,
    evidence/0,
    query_type/0,
    query_result/0,
    inference_direction/0,
    dua_error/0
]).

-endif. % DUA_HRL
