%%%-------------------------------------------------------------------
%%% @doc Supervisor module for dua application
%%% This supervises child processes within the dua app.
%%%-------------------------------------------------------------------
-module(dua_sup).

-behaviour(supervisor).

%% API
-export([start_link/0]).

%% Supervisor callbacks
-export([init/1]).

%%%===================================================================
%%% API Functions
%%%===================================================================

%% @doc Starts the supervisor and links it to the current process
-spec start_link() -> {ok, pid()} | {error, term()}.
start_link() ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, []).

%%%===================================================================
%%% Supervisor Callback
%%%===================================================================

%% @doc Init function defining child processes and supervision strategy
-spec init([]) -> {ok, {supervisor:sup_flags(), [supervisor:child_spec()]}}.
init([]) ->
    %% List child specs here. For now empty list; add your workers if any.
    Children = [
        %% Example child spec:
        %% {my_worker, {my_worker, start_link, []},
        %%  permanent, 5000, worker, [my_worker]}
    ],

    %% Choose one_for_one strategy with max 5 restarts in 10 seconds
    {ok, {{one_for_one, 5, 10}, Children}}.

