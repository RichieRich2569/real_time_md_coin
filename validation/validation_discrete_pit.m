function p = validation_discrete_pit(x, pmf, labels, u)
%VALIDATION_DISCRETE_PIT Randomised PIT for a discrete distribution.
%
%   For a discrete predictive distribution, F(X) is not uniform because the
%   CDF jumps at atoms.  The randomized PIT
%
%       P = F(x-) + U * Pr(X = x),  U ~ Uniform(0,1),
%
%   spreads each probability atom across its interval and is uniform under
%   correct calibration.

if nargin < 3 || isempty(labels)
    labels = 1:numel(pmf);
end
if nargin < 4 || isempty(u)
    u = rand;
end

pmf = pmf(:)';
labels = labels(:)';
if numel(labels) ~= numel(pmf)
    error('validation_discrete_pit:SizeMismatch', ...
        'labels and pmf must have the same number of elements.');
end

total = sum(pmf);
if total <= 0 || ~isfinite(total)
    p = NaN;
    return;
end
pmf = pmf ./ total;

mass = pmf(labels == x);
if isempty(mass)
    mass = 0;
end
fBefore = sum(pmf(labels < x));
p = min(max(fBefore + u .* mass(1), 0), 1);
end
