function n = binomialSample(~, trials, prob)
%BINOMIALSAMPLE Draw a single binomial count by Bernoulli summation.
%   n = binomialSample(obj, trials, prob) returns one draw n ~ Binomial(trials,
%   prob), the number of successes in TRIALS independent Bernoulli(PROB) trials.
%   It is computed directly as sum(rand(1, trials) < prob). The degenerate
%   cases are short-circuited: trials <= 0 or prob <= 0 gives 0, and prob >= 1
%   gives trials, so no random numbers are consumed on those paths.
%
%   The OBJ argument is unused (stateless helper exposed as a class method).
%
%   Inputs:
%     trials  Real numeric scalar number of Bernoulli trials.
%     prob    Real numeric scalar success probability.
%
%   Output:
%     n       Scalar integer count in [0, trials].
%
%   Performance note: the Bernoulli-summation path is O(trials) in time and
%   memory (it materialises a 1-by-trials random vector). The sole caller,
%   sampleGlobalTransitionProbabilities, passes small table counts so this is
%   not a bottleneck; an inverse-CDF or normal-approximation draw would remove
%   the O(trials) cost if larger counts ever arise (reported, not implemented).
%
%   See also gammaSample, sampleGlobalTransitionProbabilities.

    if ~isnumeric(trials) || ~isscalar(trials) || ~isreal(trials)
        error("RealTimeCOIN:binomialSample:invalidTrials", ...
            "trials must be a real numeric scalar.");
    end
    if ~isnumeric(prob) || ~isscalar(prob) || ~isreal(prob)
        error("RealTimeCOIN:binomialSample:invalidProb", ...
            "prob must be a real numeric scalar.");
    end

    if trials <= 0 || prob <= 0
        n = 0;
    elseif prob >= 1
        n = trials;
    else
        n = sum(rand(1, trials) < prob);
    end
end
