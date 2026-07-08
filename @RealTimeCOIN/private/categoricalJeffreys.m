function d = categoricalJeffreys(obj, p, q)
%CATEGORICALJEFFREYS Jeffreys divergence between two categorical distributions.
%
%   d = categoricalJeffreys(obj, p, q) returns the symmetric Jeffreys
%   divergence between categorical probability vectors p and q,
%       d = sum((p - q) .* log(p ./ q)) >= 0,
%   i.e. KL(p||q) + KL(q||p). Shorter of p, q is zero-padded to the common
%   length and both are renormalised (normalizeProbability) before comparison.
%   Used to compare per-context transition/global-probability rows. This is the
%   discrete analogue of the Gaussian Jeffreys helpers.

    n = max(numel(p), numel(q));
    p(end+1:n) = 0;  % zero-pad the shorter vector to a common support
    q(end+1:n) = 0;
    p = obj.normalizeProbability(p);
    q = obj.normalizeProbability(q);
    d = sum((p - q) .* log(p ./ q));
    d = jeffreysFiniteClip(d);   % finite sentinel + non-negativity clip
end
