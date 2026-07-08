function b = betaSample(obj, a, bpar)
%BETASAMPLE Draw beta random variates via the gamma ratio construction.
%   b = betaSample(obj, a, bpar) returns independent draws from the beta
%   distribution, b(i) ~ Beta(a(i), bpar(i)), using the standard identity
%   Beta(a, b) = X / (X + Y) with X ~ Gamma(a, 1) and Y ~ Gamma(b, 1) drawn by
%   gammaSample. A and BPAR are element-wise shape (concentration) parameters;
%   they must broadcast to a common size (equal-size arrays, or one of them
%   scalar). The result B has that common size.
%
%   Degenerate fallback. When both gamma draws are zero (which happens when the
%   corresponding a and bpar are both non-positive, so X = Y = 0 and the ratio
%   0/0 is undefined) the entry is set to 1. This choice is deliberately
%   ASYMMETRIC - it could equally be 0 - and is kept because the sole callers
%   are stick-breaking weights (sampleContext / instantiateCueIfNeeded) drawn
%   as Beta(1, gamma): there a = 1 > 0 in normal use, so the fallback never
%   fires on the valid path and its exact value is immaterial. Returning 1
%   keeps any degenerate stick weight from spuriously halting the break.
%
%   Inputs:
%     a     Real numeric array of first (alpha) shape parameters.
%     bpar  Real numeric array of second (beta) shape parameters.
%
%   Output:
%     b     Beta draws, size = broadcast size of A and BPAR, in [0, 1].
%
%   See also gammaSample, dirichletSample.

    if ~isnumeric(a) || ~isreal(a) || ~isnumeric(bpar) || ~isreal(bpar)
        error("RealTimeCOIN:betaSample:invalidShape", ...
            "a and bpar must be real numeric arrays.");
    end

    x = obj.gammaSample(a);
    y = obj.gammaSample(bpar);
    denom = x + y;
    b = zeros(size(denom));
    good = denom > 0;
    b(good) = x(good) ./ denom(good);
    % Asymmetric 0/0 fallback (see help): both gamma draws vanished, so assign
    % the beta variate to 1 rather than 0.
    b(~good) = 1;
end
