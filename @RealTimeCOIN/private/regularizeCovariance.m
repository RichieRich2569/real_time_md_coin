function covar = regularizeCovariance(~, covar)
%REGULARIZECOVARIANCE Symmetrise and condition a covariance matrix.
%
%   covar = regularizeCovariance(obj, covar) sanitises a possibly ill-formed
%   particle covariance so downstream inversion/factorisation is safe: it zeroes
%   non-finite entries, symmetrises, and adds diagonal loading to guarantee
%   positive definiteness and a workable condition number. Used before
%   safeInverse in the Jeffreys/Kalman MD paths.

    % Delegate to the shared PD-repair utility; the "load" tactic is a verbatim
    % copy of the zero-nonfinite / symmetrise / diagonal-loading body this
    % function used to inline.
    covar = ensurePD("load", covar);
end
