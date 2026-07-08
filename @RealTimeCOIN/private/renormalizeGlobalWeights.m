function w = renormalizeGlobalWeights(w)
%RENORMALIZEGLOBALWEIGHTS Zero non-finite entries and renormalise to sum one.
%
%   w = renormalizeGlobalWeights(w) replaces any non-finite entry of w with zero
%   and, if the remaining entries have a strictly positive sum, scales w so that
%   it sums to one. An all-zero (or all-non-finite) input is returned unchanged
%   rather than forced to a uniform distribution.
%
%   Shared by the global_*_probabilities read-outs so both apply the exact same
%   guarded normalisation. Unlike normalizeProbability, there is no uniform
%   fallback and no realmin flooring, so an empty franchise stays all-zero.
    w(~isfinite(w)) = 0;
    s = sum(w);
    if s > 0
        w = w ./ s;
    end
end
