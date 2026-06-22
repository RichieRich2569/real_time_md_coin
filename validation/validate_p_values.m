function results = validate_p_values(varargin)
%VALIDATE_P_VALUES Posterior predictive p-value validation for RealTimeCOIN.
%
%   results = validate_p_values('NumDatasets', 100, 'Trials', 200)
%   generates scalar synthetic streams, scores each cue/feedback before it
%   is consumed, and reports Kolmogorov-Smirnov style distances from a
%   uniform distribution. This is validation code, not a fast unit test.
%
%   This compatibility wrapper now delegates to the extended validator,
%   which keeps the original feedback/cue fields and adds state and
%   parameter-rank diagnostics.

results = validate_p_values_extended(varargin{:});
end
