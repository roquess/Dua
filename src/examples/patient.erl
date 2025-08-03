%%%-------------------------------------------------------------------
%%% @doc Enhanced Medical Diagnosis System with Combined Prior and DUA Backward Inference
%%%
%%% This version improves backward inference results by incorporating explicit priors
%%% with DUA likelihood estimates, producing more accurate disease posteriors.
%%%-------------------------------------------------------------------

-module(patient).
-include("dua.hrl").
-export([run_example/0]).

run_example() ->
    io:format("=== PATIENT DIAGNOSIS SYSTEM with IMPROVED BACKWARD INFERENCE ===~n~n"),

    {ok, Graph} = create_medical_graph(),
    validate_graph_structure(Graph),
    run_test_scenarios(Graph).

create_medical_graph() ->
    Graph0 = dua_graph:new(#{description => "Extended medical diagnosis" }),

    Flu = dua_node:new_boolean(flu),
    Covid = dua_node:new_boolean(covid),
    Allergy = dua_node:new_boolean(allergy),
    Pneumonia = dua_node:new_boolean(pneumonia),
    Bronchitis = dua_node:new_boolean(bronchitis),

    Fever = dua_node:new_boolean(fever),
    Fatigue = dua_node:new_boolean(fatigue),
    LossSmell = dua_node:new_boolean(loss_of_smell),
    Cough = dua_node:new_boolean(cough),
    ShortBreath = dua_node:new_boolean(shortness_of_breath),
    Headache = dua_node:new_boolean(headache),
    RunnyNose = dua_node:new_boolean(runny_nose),

    {ok, G1} = dua_graph:add_node(Graph0, flu, Flu),
    {ok, G2} = dua_graph:add_node(G1, covid, Covid),
    {ok, G3} = dua_graph:add_node(G2, allergy, Allergy),
    {ok, G4} = dua_graph:add_node(G3, pneumonia, Pneumonia),
    {ok, G5} = dua_graph:add_node(G4, bronchitis, Bronchitis),
    {ok, G6} = dua_graph:add_node(G5, fever, Fever),
    {ok, G7} = dua_graph:add_node(G6, fatigue, Fatigue),
    {ok, G8} = dua_graph:add_node(G7, loss_of_smell, LossSmell),
    {ok, G9} = dua_graph:add_node(G8, cough, Cough),
    {ok, G10} = dua_graph:add_node(G9, shortness_of_breath, ShortBreath),
    {ok, G11} = dua_graph:add_node(G10, headache, Headache),
    {ok, GraphFinal} = dua_graph:add_node(G11, runny_nose, RunnyNose),

    % Add edges with conditional probabilities
    {ok, G12} = dua_graph:add_edge(GraphFinal, flu, fever, 0.85),
    {ok, G13} = dua_graph:add_edge(G12, flu, fatigue, 0.75),
    {ok, G14} = dua_graph:add_edge(G13, flu, cough, 0.60),
    {ok, G15} = dua_graph:add_edge(G14, flu, headache, 0.50),

    {ok, G16} = dua_graph:add_edge(G15, covid, fever, 0.90),
    {ok, G17} = dua_graph:add_edge(G16, covid, fatigue, 0.85),
    {ok, G18} = dua_graph:add_edge(G17, covid, cough, 0.70),
    {ok, G19} = dua_graph:add_edge(G18, covid, loss_of_smell, 0.75),
    {ok, G20} = dua_graph:add_edge(G19, covid, shortness_of_breath, 0.60),

    {ok, G21} = dua_graph:add_edge(G20, allergy, runny_nose, 0.80),
    {ok, G22} = dua_graph:add_edge(G21, allergy, fatigue, 0.40),
    {ok, G23} = dua_graph:add_edge(G22, allergy, headache, 0.30),

    {ok, G24} = dua_graph:add_edge(G23, pneumonia, fever, 0.95),
    {ok, G25} = dua_graph:add_edge(G24, pneumonia, fatigue, 0.80),
    {ok, G26} = dua_graph:add_edge(G25, pneumonia, cough, 0.90),
    {ok, G27} = dua_graph:add_edge(G26, pneumonia, shortness_of_breath, 0.80),

    {ok, G28} = dua_graph:add_edge(G27, bronchitis, cough, 0.85),
    {ok, G29} = dua_graph:add_edge(G28, bronchitis, fatigue, 0.70),
    {ok, GraphFinalComplete} = dua_graph:add_edge(G29, bronchitis, shortness_of_breath, 0.60),

    {ok, GraphFinalComplete}.


validate_graph_structure(Graph) ->
    Nodes = Graph#dua_graph.nodes,
    Edges = Graph#dua_graph.edges,
    io:format("~n=== GRAPH STRUCTURE VALIDATION ===~n"),
    io:format("Nodes count: ~p~n", [maps:size(Nodes)]),
    io:format("Edges count: ~p~n", [length(Edges)]),
    lists:foreach(fun(#dua_edge{from=F,to=T,weight=W}) ->
        io:format("  ~p -> ~p (weight=~.2f)~n", [F,T,W])
    end, Edges),
    ok.


run_test_scenarios(Graph) ->
    Scenarios = [
        #{desc => "Fever only", evidence => #{fever => true}},
        #{desc => "Fever + cough + fatigue", evidence => #{fever => true,cough => true,fatigue => true}},
        #{desc => "COVID symptoms: fever, loss of smell, shortness of breath", evidence => #{fever => true,loss_of_smell => true,shortness_of_breath => true}},
        #{desc => "Allergy symptoms: runny nose and headache", evidence => #{runny_nose => true,headache => true}},
        #{desc => "Bronchitis symptoms: cough and shortness of breath", evidence => #{cough => true,shortness_of_breath => true}},
        #{desc => "Pneumonia symptoms: cough, fever and shortness of breath", evidence => #{cough => true,fever => true,shortness_of_breath => true}},
        #{desc => "Fatigue only", evidence => #{fatigue => true}}
    ],
    lists:foreach(fun(S) ->
        run_single_scenario(Graph, maps:get(evidence,S), maps:get(desc,S))
    end, Scenarios).


run_single_scenario(Graph, Evidence, Description) ->
    io:format("~n--- Scenario: ~s ---~n", [Description]),
    io:format("Evidence: ~p~n", [Evidence]),

    case dua_backward:infer(Graph, Evidence) of
        {ok, Result} ->
            io:format("DUA Backward Inference Result:~n"),
            print_inference_result(Result);
        Error ->
            io:format("DUA backward inference failed: ~p~n", [Error])
    end.

print_inference_result(#{beliefs := Beliefs, confidence := Confidence, algorithm := Algorithm}) ->
    io:format("Algorithm: ~p, Confidence: ~.2f~n", [Algorithm, Confidence]),
    Diseases = [flu,covid,allergy,pneumonia,bronchitis],
    Priors = #{
        flu => 0.12,
        covid => 0.08,
        allergy => 0.25,
        pneumonia => 0.03,
        bronchitis => 0.05
    },

    io:format("Disease probabilities after combining DUA likelihood and priors:~n"),
    lists:foreach(fun(Disease) ->
        case maps:get(Disease, Beliefs, undefined) of
            undefined -> io:format("  ~p: no belief data~n", [Disease]);
            Belief ->
                case dua_belief:most_likely_state(Belief) of
                    {ok, {true, Likelihood}} ->
                        Prior = maps:get(Disease, Priors, 0.01),
                        Posterior = compute_posterior(Prior, Likelihood),
                        io:format("  ~p: Prior=~.3f, Likelihood=~.3f, Posterior=~.3f~n",
                                  [Disease, Prior, Likelihood, Posterior]);
                    {ok, {false, _}} ->
                        io:format("  ~p: mostly FALSE belief~n", [Disease]);
                    _ ->
                        io:format("  ~p: error interpreting belief~n", [Disease])
                end
        end
    end, Diseases).

%% Bayesian update: P(Disease|Evidence) proportional to P(Evidence|Disease)*P(Disease)
compute_posterior(Prior, Likelihood) when Prior >= 0, Likelihood >= 0 ->
    NotPrior = 1.0 - Prior,
    NotLikelihood = max(0.01, 1.0 - Likelihood),
    Numerator = Prior * Likelihood,
    Denominator = Numerator + NotPrior * NotLikelihood,
    case Denominator of
        +0.0 -> +0.0;
        _ -> Numerator / Denominator
    end.
