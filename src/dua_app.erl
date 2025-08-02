-module(dua_app).
-behaviour(application).

-export([start/2, stop/1]).

start(_Type, _Args) ->
    lager:start(),
    dua_sup:start_link().

stop(_State) ->
    lager:stop(),
    ok.

