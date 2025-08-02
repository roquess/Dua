%%%-------------------------------------------------------------------
%%% @doc DUA Probability Calculations Module
%%%
%%% This module provides core probability operations, distributions,
%%% and statistical functions used throughout the DUA system.
%%%-------------------------------------------------------------------

-module(dua_probability).

-include("dua.hrl").

-export([
    %% Basic probability operations
    multiply/2,
    add/2,
    normalize_vector/1,

    %% Conditional probability
    bayes_rule/3,
    conditional_probability/3,
    joint_probability/1,  %% Correct arity is /1

    %% Distributions
    uniform_distribution/1,
    gaussian_distribution/2,
    beta_distribution/2,

    %% Statistical measures
    expectation/1,
    variance/1,
    covariance/2,
    correlation/2,

    %% Information theory
    entropy/1,
    mutual_information/2,
    kl_divergence/2,
    cross_entropy/2,

    %% Sampling
    sample_from_distribution/1,
    sample_n/2,
    importance_sampling/3,

    %% Validation
    is_valid_probability/1,
    is_valid_distribution/1,

    %% Utility functions
    log_probability/1,
    exp_probability/1,
    safe_divide/2
]).

%%%===================================================================
%%% Basic Probability Operations
%%%===================================================================

%% @doc Multiply two probabilities
-spec multiply(probability(), probability()) -> probability().
multiply(P1, P2) when P1 >= 0.0, P1 =< 1.0, P2 >= 0.0, P2 =< 1.0 ->
    P1 * P2;
multiply(P1, P2) ->
    error({invalid_probabilities, P1, P2}).

%% @doc Add two probabilities (for mutually exclusive events)
-spec add(probability(), probability()) -> probability().
add(P1, P2) when P1 >= 0.0, P1 =< 1.0, P2 >= 0.0, P2 =< 1.0 ->
    min(P1 + P2, 1.0);
add(P1, P2) ->
    error({invalid_probabilities, P1, P2}).

%% @doc Normalize a vector of probabilities
-spec normalize_vector([probability()]) -> [probability()].
normalize_vector([]) ->
    [];
normalize_vector(Probabilities) ->
    Sum = lists:sum(Probabilities),
    case Sum > ?DEFAULT_PRECISION of
        true ->
            [P / Sum || P <- Probabilities];
        false ->
            %% If all probabilities are essentially zero, make uniform
            Uniform = 1.0 / length(Probabilities),
            lists:duplicate(length(Probabilities), Uniform)
    end.

%%%===================================================================
%%% Conditional Probability
%%%===================================================================

%% @doc Apply Bayes' rule: P(A|B) = P(B|A) * P(A) / P(B)
-spec bayes_rule(probability(), probability(), probability()) -> probability().
bayes_rule(PB_given_A, PA, PB) when PB > ?DEFAULT_PRECISION ->
    (PB_given_A * PA) / PB;
bayes_rule(_PB_given_A, _PA, _PB) ->
    0.0.

%% @doc Calculate conditional probability from joint and marginal
-spec conditional_probability(probability(), probability(), probability()) -> probability().
conditional_probability(PJoint, PMarginal, _PCondition) when PMarginal > ?DEFAULT_PRECISION ->
    PJoint / PMarginal;
conditional_probability(_PJoint, _PMarginal, _PCondition) ->
    0.0.

%% @doc Calculate joint probability of independent events
-spec joint_probability([probability()]) -> probability().
joint_probability([]) ->
    1.0;
joint_probability(Probabilities) ->
    lists:foldl(fun multiply/2, 1.0, Probabilities).

%%%===================================================================
%%% Distributions
%%%===================================================================

%% @doc Create uniform distribution over N outcomes
-spec uniform_distribution(pos_integer()) -> [probability()].
uniform_distribution(N) when N > 0 ->
    Prob = 1.0 / N,
    lists:duplicate(N, Prob).

%% @doc Generate Gaussian (normal) distribution approximation
-spec gaussian_distribution(float(), float()) -> fun((float()) -> probability()).
gaussian_distribution(Mean, StdDev) when StdDev > 0.0 ->
    fun(X) ->
        Exponent = -0.5 * math:pow((X - Mean) / StdDev, 2),
        (1.0 / (StdDev * math:sqrt(2 * math:pi()))) * math:exp(Exponent)
    end;
gaussian_distribution(_Mean, _StdDev) ->
    error(invalid_standard_deviation).

%% @doc Generate Beta distribution approximation
-spec beta_distribution(float(), float()) -> fun((probability()) -> float()).
beta_distribution(Alpha, Beta) when Alpha > 0.0, Beta > 0.0 ->
    fun(X) when X >= 0.0, X =< 1.0 ->
        math:pow(X, Alpha - 1) * math:pow(1 - X, Beta - 1) / beta_function(Alpha, Beta);
    (_X) ->
        0.0
    end;
beta_distribution(_Alpha, _Beta) ->
    error(invalid_beta_parameters).

%%%===================================================================
%%% Statistical Measures
%%%===================================================================

%% @doc Calculate expectation (mean) of a probability distribution
-spec expectation(#{term() => probability()}) -> float().
expectation(Distribution) ->
    maps:fold(fun(Value, Prob, Acc) ->
        NumValue = case is_number(Value) of
            true -> Value;
            false -> value_to_number(Value)
        end,
        Acc + NumValue * Prob
    end, 0.0, Distribution).

%% @doc Calculate variance of a probability distribution
-spec variance(#{term() => probability()}) -> float().
variance(Distribution) ->
    Mean = expectation(Distribution),
    maps:fold(fun(Value, Prob, Acc) ->
        NumValue = case is_number(Value) of
            true -> Value;
            false -> value_to_number(Value)
        end,
        Acc + Prob * math:pow(NumValue - Mean, 2)
    end, 0.0, Distribution).

%% @doc Calculate covariance between two distributions
-spec covariance(#{term() => probability()}, #{term() => probability()}) -> float().
covariance(Dist1, Dist2) ->
    Mean1 = expectation(Dist1),
    Mean2 = expectation(Dist2),
    Keys1 = sets:from_list(maps:keys(Dist1)),
    Keys2 = sets:from_list(maps:keys(Dist2)),
    CommonKeys = sets:to_list(sets:intersection(Keys1, Keys2)),
    lists:foldl(fun(Key, Acc) ->
        Prob1 = maps:get(Key, Dist1),
        Prob2 = maps:get(Key, Dist2),
        Value1 = case is_number(Key) of true -> Key; false -> value_to_number(Key) end,
        Value2 = Value1,
        Acc + (Value1 - Mean1) * (Value2 - Mean2) * min(Prob1, Prob2)
    end, 0.0, CommonKeys).

%% @doc Calculate correlation coefficient between two distributions
-spec correlation(#{term() => probability()}, #{term() => probability()}) -> float().
correlation(Dist1, Dist2) ->
    Cov = covariance(Dist1, Dist2),
    Var1 = variance(Dist1),
    Var2 = variance(Dist2),
    case Var1 * Var2 of
        Denominator when Denominator > ?DEFAULT_PRECISION ->
            Cov / math:sqrt(Denominator);
        _ ->
            0.0
    end.

%%%===================================================================
%%% Information Theory
%%%===================================================================

%% @doc Calculate entropy of a probability distribution
-spec entropy(#{term() => probability()}) -> float().
entropy(Distribution) ->
    maps:fold(fun(_Value, Prob, Acc) ->
        case Prob > ?DEFAULT_PRECISION of
            true ->
                Acc - Prob * log2(Prob);
            false ->
                Acc
        end
    end, 0.0, Distribution).

%% @doc Calculate mutual information between two distributions
-spec mutual_information(#{term() => probability()}, #{term() => probability()}) -> float().
mutual_information(Dist1, Dist2) ->
    %% I(X;Y) = H(X) + H(Y) - H(X,Y)
    H_X = entropy(Dist1),
    H_Y = entropy(Dist2),
    H_XY = joint_entropy(Dist1, Dist2),
    H_X + H_Y - H_XY.

%% @doc Calculate KL divergence between two distributions
-spec kl_divergence(#{term() => probability()}, #{term() => probability()}) -> float().
kl_divergence(P, Q) ->
    %% KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
    CommonKeys = sets:to_list(
        sets:intersection(
            sets:from_list(maps:keys(P)),
            sets:from_list(maps:keys(Q))
        )
    ),
    lists:foldl(fun(Key, Acc) ->
        PProb = maps:get(Key, P),
        QProb = maps:get(Key, Q, ?DEFAULT_PRECISION),
        case PProb > ?DEFAULT_PRECISION andalso QProb > ?DEFAULT_PRECISION of
            true ->
                Acc + PProb * log2(PProb / QProb);
            false ->
                Acc
        end
    end, 0.0, CommonKeys).

%% @doc Calculate cross entropy between two distributions
-spec cross_entropy(#{term() => probability()}, #{term() => probability()}) -> float().
cross_entropy(P, Q) ->
    %% H(P,Q) = -Σ P(x) * log(Q(x))
    CommonKeys = sets:to_list(
        sets:intersection(
            sets:from_list(maps:keys(P)),
            sets:from_list(maps:keys(Q))
        )
    ),
    lists:foldl(fun(Key, Acc) ->
        PProb = maps:get(Key, P),
        QProb = maps:get(Key, Q, ?DEFAULT_PRECISION),
        case PProb > ?DEFAULT_PRECISION andalso QProb > ?DEFAULT_PRECISION of
            true ->
                Acc - PProb * log2(QProb);
            false ->
                Acc
        end
    end, 0.0, CommonKeys).

%%%===================================================================
%%% Sampling
%%%===================================================================

%% @doc Sample from a probability distribution
-spec sample_from_distribution(#{term() => probability()}) -> term().
sample_from_distribution(Distribution) ->
    Random = rand:uniform(),
    sample_from_cumulative(maps:to_list(Distribution), Random, 0.0).

%% @doc Sample N times from a distribution
-spec sample_n(#{term() => probability()}, pos_integer()) -> [term()].
sample_n(Distribution, N) ->
    [sample_from_distribution(Distribution) || _ <- lists:seq(1, N)].

%% @doc Importance sampling with proposal distribution
-spec importance_sampling(#{term() => probability()}, #{term() => probability()}, pos_integer()) ->
    [{term(), float()}].
importance_sampling(Target, Proposal, NumSamples) ->
    Samples = sample_n(Proposal, NumSamples),
    lists:map(fun(Sample) ->
        TargetProb = maps:get(Sample, Target, 0.0),
        ProposalProb = maps:get(Sample, Proposal, ?DEFAULT_PRECISION),
        Weight = safe_divide(TargetProb, ProposalProb),
        {Sample, Weight}
    end, Samples).

%%%===================================================================
%%% Validation
%%%===================================================================

%% @doc Check if a value is a valid probability
-spec is_valid_probability(term()) -> boolean().
is_valid_probability(P) when is_number(P) ->
    P >= 0.0 andalso P =< 1.0;
is_valid_probability(_) ->
    false.

%% @doc Check if a distribution is valid
-spec is_valid_distribution(#{term() => probability()}) -> boolean().
is_valid_distribution(Distribution) ->
    %% Check all probabilities are valid
    AllValid = maps:fold(fun(_Key, Prob, Valid) ->
        Valid andalso is_valid_probability(Prob)
    end, true, Distribution),
    %% Check probabilities sum to approximately 1.0
    Sum = maps:fold(fun(_, Prob, Acc) -> Acc + Prob end, 0.0, Distribution),
    SumValid = abs(Sum - 1.0) < ?DEFAULT_PRECISION,
    AllValid andalso SumValid.

%%%===================================================================
%%% Utility Functions
%%%===================================================================

%% @doc Convert probability to log space (for numerical stability)
-spec log_probability(probability()) -> float().
log_probability(P) when P > 0.0 ->
    math:log(P);
log_probability(_P) ->
    -math:log(0.0).  % -infinity

%% @doc Convert from log space back to probability
-spec exp_probability(float()) -> probability().
exp_probability(LogP) ->
    max(0.0, min(1.0, math:exp(LogP))).

%% @doc Safe division with default value for division by zero
-spec safe_divide(float(), float()) -> float().
safe_divide(_Numerator, Denominator) when abs(Denominator) < ?DEFAULT_PRECISION ->
    0.0;
safe_divide(Numerator, Denominator) ->
    Numerator / Denominator.

%%%===================================================================
%%% Internal Functions
%%%===================================================================

%% @private
value_to_number(true) -> 1.0;
value_to_number(false) -> 0.0;
value_to_number(Value) when is_atom(Value) ->
    %% Simple hash-based conversion for atoms
    Hash = erlang:phash2(Value),
    Hash / 1000000.0;
value_to_number(Value) when is_binary(Value) ->
    Hash = erlang:phash2(Value),
    Hash / 1000000.0;
value_to_number(_) -> 0.0.

%% @private
log2(X) when X > 0.0 ->
    math:log(X) / math:log(2);
log2(_X) ->
    0.0.

%% @private
sample_from_cumulative([], _Random, _CumProb) ->
    undefined;
sample_from_cumulative([{Value, Prob} | Rest], Random, CumProb) ->
    NewCumProb = CumProb + Prob,
    case Random =< NewCumProb of
        true -> Value;
        false -> sample_from_cumulative(Rest, Random, NewCumProb)
    end.

%% @private
joint_entropy(Dist1, Dist2) ->
    %% Simplified joint entropy calculation. In practice, you would use the actual joint distribution.
    Keys1 = sets:from_list(maps:keys(Dist1)),
    Keys2 = sets:from_list(maps:keys(Dist2)),
    CommonKeys = sets:to_list(sets:intersection(Keys1, Keys2)),
    lists:foldl(fun(Key, Acc) ->
        Prob1 = maps:get(Key, Dist1),
        Prob2 = maps:get(Key, Dist2),
        JointProb = min(Prob1, Prob2),  %% Simplified assumption
        case JointProb > ?DEFAULT_PRECISION of
            true -> Acc - JointProb * log2(JointProb);
            false -> Acc
        end
    end, 0.0, CommonKeys).

%% @private
beta_function(Alpha, Beta) ->
    %% Simplified beta function calculation using log gamma for computational efficiency
    math:exp(log_gamma(Alpha) + log_gamma(Beta) - log_gamma(Alpha + Beta)).

%% @private
log_gamma(X) when X > 0.0 ->
    %% Stirling's approximation for log(Γ(x))
    (X - 0.5) * math:log(X) - X + 0.5 * math:log(2 * math:pi()).

