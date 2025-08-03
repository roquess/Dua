%%%-------------------------------------------------------------------
%%% @doc Simple Car Diagnosis - Backward Inference Example
%%%
%%% A minimal example showing backward inference for car troubleshooting:
%%% - Observe symptoms (car problems)
%%% - Infer probable causes (mechanical issues)
%%%-------------------------------------------------------------------

-module(car_diagnosis).
-include("dua.hrl").
-export([run_example/0]).

run_example() ->
    io:format("=== SIMPLE CAR DIAGNOSIS: BACKWARD INFERENCE ===~n~n"),
    
    % Create a simple car diagnostic graph
    {ok, Graph} = create_car_graph(),
    
    % Show the graph structure
    display_car_model(Graph),
    
    % Test with different observed symptoms
    test_car_scenarios(Graph).

%% Create simple graph: 3 car problems -> 4 symptoms
create_car_graph() ->
    Graph0 = dua_graph:new(#{description => "Car diagnostic system"}),
    
    % === CAUSES (Car Problems) ===
    DeadBattery = dua_node:new_boolean(dead_battery),
    EmptyTank = dua_node:new_boolean(empty_tank),
    BadStarter = dua_node:new_boolean(bad_starter),
    
    % === EFFECTS (Symptoms) ===
    NoStart = dua_node:new_boolean(wont_start),
    DimLights = dua_node:new_boolean(dim_lights),
    ClickSound = dua_node:new_boolean(clicking_sound),
    NoFuelGauge = dua_node:new_boolean(fuel_gauge_empty),
    
    % Add nodes to graph
    {ok, G1} = dua_graph:add_node(Graph0, dead_battery, DeadBattery),
    {ok, G2} = dua_graph:add_node(G1, empty_tank, EmptyTank),
    {ok, G3} = dua_graph:add_node(G2, bad_starter, BadStarter),
    {ok, G4} = dua_graph:add_node(G3, wont_start, NoStart),
    {ok, G5} = dua_graph:add_node(G4, dim_lights, DimLights),
    {ok, G6} = dua_graph:add_node(G5, clicking_sound, ClickSound),
    {ok, G7} = dua_graph:add_node(G6, fuel_gauge_empty, NoFuelGauge),
    
    % === CAUSAL RELATIONSHIPS ===
    % Dead battery effects
    {ok, G8} = dua_graph:add_edge(G7, dead_battery, wont_start, 0.95),      % Dead battery almost always prevents starting
    {ok, G9} = dua_graph:add_edge(G8, dead_battery, dim_lights, 0.90),      % Dead battery causes dim lights
    {ok, G10} = dua_graph:add_edge(G9, dead_battery, clicking_sound, 0.80), % Dead battery often causes clicking
    {ok, G11} = dua_graph:add_edge(G10, dead_battery, fuel_gauge_empty, 0.10), % Rarely affects fuel gauge
    
    % Empty tank effects
    {ok, G12} = dua_graph:add_edge(G11, empty_tank, wont_start, 0.90),       % Empty tank prevents starting
    {ok, G13} = dua_graph:add_edge(G12, empty_tank, dim_lights, 0.15),       % Rarely affects lights
    {ok, G14} = dua_graph:add_edge(G13, empty_tank, clicking_sound, 0.20),   % Rarely causes clicking
    {ok, G15} = dua_graph:add_edge(G14, empty_tank, fuel_gauge_empty, 0.85), % Usually shows on fuel gauge
    
    % Bad starter effects
    {ok, G16} = dua_graph:add_edge(G15, bad_starter, wont_start, 0.85),      % Bad starter prevents starting
    {ok, G17} = dua_graph:add_edge(G16, bad_starter, dim_lights, 0.25),      % Sometimes affects electrical
    {ok, G18} = dua_graph:add_edge(G17, bad_starter, clicking_sound, 0.70),  % Often causes clicking
    {ok, GraphFinal} = dua_graph:add_edge(G18, bad_starter, fuel_gauge_empty, 0.05), % Rarely affects fuel gauge
    
    {ok, GraphFinal}.

display_car_model(Graph) ->
    io:format("~n=== CAR DIAGNOSTIC MODEL ===~n"),
    io:format("POSSIBLE CAUSES:~n"),
    io:format("  - dead_battery (battery is dead)~n"),
    io:format("  - empty_tank (out of fuel)~n"),
    io:format("  - bad_starter (starter motor broken)~n"),
    
    io:format("~nOBSERVABLE SYMPTOMS:~n"),
    io:format("  - wont_start (engine won't start)~n"),
    io:format("  - dim_lights (headlights are dim)~n"),
    io:format("  - clicking_sound (clicking when turning key)~n"),
    io:format("  - fuel_gauge_empty (fuel gauge shows empty)~n"),
    
    io:format("~nCAUSAL RELATIONSHIPS:~n"),
    lists:foreach(fun(#dua_edge{from=F, to=T, weight=W}) ->
        Percentage = round(W * 100),
        io:format("  ~p -> ~p (~p% chance)~n", [F, T, Percentage])
    end, Graph#dua_graph.edges).

test_car_scenarios(Graph) ->
    io:format("~n=== DIAGNOSTIC SCENARIOS ===~n"),
    
    % Scenario 1: Car won't start only
    diagnose_car(Graph, #{wont_start => true}, "Car won't start"),
    
    % Scenario 2: Dim lights only  
    diagnose_car(Graph, #{dim_lights => true}, "Headlights are dim"),
    
    % Scenario 3: Clicking sound only
    diagnose_car(Graph, #{clicking_sound => true}, "Clicking sound when turning key"),
    
    % Scenario 4: Empty fuel gauge only
    diagnose_car(Graph, #{fuel_gauge_empty => true}, "Fuel gauge shows empty"),
    
    % Scenario 5: Won't start + dim lights
    diagnose_car(Graph, #{wont_start => true, dim_lights => true}, "Won't start + dim lights"),
    
    % Scenario 6: Won't start + empty fuel gauge
    diagnose_car(Graph, #{wont_start => true, fuel_gauge_empty => true}, "Won't start + fuel gauge empty"),
    
    % Scenario 7: Won't start + clicking sound
    diagnose_car(Graph, #{wont_start => true, clicking_sound => true}, "Won't start + clicking sound").

diagnose_car(Graph, Symptoms, Description) ->
    io:format("~n--- CASE: ~s ---~n", [Description]),
    io:format("OBSERVED SYMPTOMS: ~p~n", [Symptoms]),
    
    case dua_backward:infer(Graph, Symptoms) of
        {ok, Result} ->
            io:format("~nDIAGNOSTIC RESULTS:~n"),
            show_diagnosis(Result, Symptoms);
        Error ->
            io:format("Diagnosis failed: ~p~n", [Error])
    end.

show_diagnosis(#{beliefs := Beliefs}, Symptoms) ->
    % Prior probabilities (base rates of car problems)
    Priors = #{
        dead_battery => 0.30,  % 30% chance battery is dead
        empty_tank => 0.15,    % 15% chance tank is empty  
        bad_starter => 0.08    % 8% chance starter is bad
    },
    
    io:format("MOST LIKELY CAUSES:~n"),
    
    Problems = [dead_battery, empty_tank, bad_starter],
    Results = lists:map(fun(Problem) ->
        case maps:get(Problem, Beliefs, undefined) of
            undefined -> 
                {Problem, 0.0};
            Belief ->
                % Extract true probability directly from the belief map
                TrueProbability = maps:get(true, Belief, 0.0),
                Prior = maps:get(Problem, Priors, 0.05),
                Posterior = bayesian_update(Prior, TrueProbability),
                {Problem, Posterior}
        end
    end, Problems),
    
    % Sort by probability (highest first)
    SortedResults = lists:sort(fun({_, P1}, {_, P2}) -> P1 > P2 end, Results),
    
    lists:foreach(fun({Problem, Probability}) ->
        Percentage = Probability * 100,
        Confidence = if 
            Percentage > 70 -> " (VERY LIKELY)";
            Percentage > 40 -> " (LIKELY)"; 
            Percentage > 20 -> " (POSSIBLE)";
            true -> " (UNLIKELY)"
        end,
        io:format("  ~p: ~.1f%~s~n", [Problem, Percentage, Confidence])
    end, SortedResults),
    
    io:format("~nRECOMMENDATION:~n"),
    give_recommendation(SortedResults, Symptoms).

%% Simple Bayesian calculation
bayesian_update(Prior, Likelihood) ->
    NotPrior = 1.0 - Prior,
    NotLikelihood = max(0.1, 1.0 - Likelihood),
    
    Numerator = Prior * Likelihood,
    Denominator = Numerator + NotPrior * NotLikelihood,
    
    case Denominator of
        -0.0 -> 0.0;
        +0.0 -> 0.0;
        _ -> Numerator / Denominator
    end.

give_recommendation([{TopCause, TopProb} | _], Symptoms) ->
    HasDimLights = maps:get(dim_lights, Symptoms, false),
    HasClicking = maps:get(clicking_sound, Symptoms, false),
    HasEmptyGauge = maps:get(fuel_gauge_empty, Symptoms, false),
    
    case TopCause of
        dead_battery when TopProb > 0.5 ->
            if 
                HasDimLights -> 
                    io:format("  → Check battery voltage and connections~n");
                HasClicking ->
                    io:format("  → Battery likely dead - try jump start or replace battery~n");
                true ->
                    io:format("  → Test battery and charging system~n")
            end;
        empty_tank when TopProb > 0.5 ->
            if
                HasEmptyGauge ->
                    io:format("  → Add fuel to tank - gauge indicates empty~n");
                true ->
                    io:format("  → Check fuel level manually - gauge may be faulty~n")
            end;
        bad_starter when TopProb > 0.5 ->
            if
                HasClicking ->
                    io:format("  → Starter motor likely faulty - have it tested/replaced~n");
                true ->
                    io:format("  → Check starter motor and electrical connections~n")
            end;
        _ ->
            io:format("  → Multiple causes possible - systematic diagnosis needed~n")
    end.
