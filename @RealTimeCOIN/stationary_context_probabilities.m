function p = stationary_context_probabilities(obj)
%STATIONARY_CONTEXT_PROBABILITIES Stationary distribution of the context chain.
%
%   p = stationary_context_probabilities(obj) returns the 1-by-K stationary
%   distribution of the expected local context-transition matrix in the aligned
%   global-context frame (K = context_alignment(obj).K). Mirrors COIN's
%   plot_stationary_probabilities: the novel-context column is dropped and each
%   row renormalised to form a stochastic matrix over the instantiated contexts
%   before solving for its stationary distribution. Dimension-independent.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    alignment = ensureContextAlignment(obj);
    K = alignment.K;
    if K == 0
        p = [];
        return;
    end
    T = alignment.global_contexts.transition_prob(1:K, 1:K);
    rowSum = sum(T, 2);
    zeroRows = rowSum <= 0;
    T(~zeroRows, :) = T(~zeroRows, :) ./ rowSum(~zeroRows);
    T(zeroRows, :) = 1 / K;              % uniform fallback for empty rows
    p = obj.stationary_distribution(T);
end
