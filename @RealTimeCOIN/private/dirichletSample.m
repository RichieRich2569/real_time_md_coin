function x = dirichletSample(obj, alpha)
%DIRICHLETSAMPLE Draw a Dirichlet-distributed probability vector.
%   x = dirichletSample(obj, alpha) returns a column vector x ~ Dirichlet(alpha)
%   using the gamma-normalisation construction: draw g(i) ~ Gamma(alpha(i), 1)
%   and set x = g / sum(g). ALPHA is a vector of concentration parameters (any
%   orientation; it is flattened to a column) and X has the same number of
%   elements, is non-negative and sums to 1.
%
%   Degenerate fallback. If every gamma draw is zero (sum(draws) <= 0, which can
%   occur when all positive-alpha entries happen to draw 0, or when no entry has
%   a positive alpha at all) the routine places all probability mass on the
%   first entry with alpha > 0, or on entry 1 if none is positive. This keeps X
%   a valid probability vector instead of returning 0/0 = NaN.
%
%   Inputs:
%     alpha  Real numeric vector of Dirichlet concentration parameters.
%
%   Output:
%     x      Column probability vector, numel(alpha)-by-1, summing to 1.
%
%   See also gammaSample, betaSample, sampleGlobalCueProbabilities.

    if ~isnumeric(alpha) || ~isreal(alpha) || ~isvector(alpha)
        error("RealTimeCOIN:dirichletSample:invalidAlpha", ...
            "alpha must be a real numeric vector.");
    end

    alpha = alpha(:);
    draws = obj.gammaSample(alpha);
    if sum(draws) <= 0
        % All gamma draws vanished: concentrate mass on the first positive-alpha
        % component (or component 1) so the output stays a valid distribution.
        draws = zeros(size(alpha));
        first = find(alpha > 0, 1);
        if isempty(first)
            first = 1;
        end
        draws(first) = 1;
    end
    x = draws ./ sum(draws);
end
