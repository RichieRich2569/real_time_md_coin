function [mu, v] = validation_predictive_feedback_moments(coin, q)
%VALIDATION_PREDICTIVE_FEEDBACK_MOMENTS One-step feedback moments.
%
%   This mirrors RealTimeCOIN's public predictive CDF calculation using
%   values exposed by diagnostics().  It is deliberately kept in validation
%   code so the production class does not need extra accessors just for
%   scientific checks.

if nargin < 2
    q = [];
end

diag = coin.diagnostics();
raw = diag.raw;
Cmax = coin.max_contexts + 1;
P = coin.num_particles;

prior = zeros(Cmax, P);
for pIdx = 1:P
    prior(:, pIdx) = raw.local_transition_matrix(raw.context(pIdx), :, pIdx)';
end
prior = normalize_columns(prior);

if isempty(q)
    weights = prior;
else
    qCol = min(q, size(raw.local_cue_matrix, 2));
    cueLikelihood = squeeze(raw.local_cue_matrix(:, qCol, :));
    if P == 1
        cueLikelihood = cueLikelihood(:);
    end
    weights = normalize_columns(prior .* cueLikelihood);
end

stateMean = raw.retention .* raw.state_filtered_mean + raw.drift;
stateVar = raw.retention.^2 .* raw.state_filtered_var + coin.sigma_process_noise^2;
for pIdx = 1:P
    novel = min(raw.C(pIdx) + 1, Cmax);
    if raw.C(pIdx) < coin.max_contexts
        a = raw.retention(novel, pIdx);
        d = raw.drift(novel, pIdx);
        stateMean(novel, pIdx) = d ./ max(1 - a, eps);
        stateVar(novel, pIdx) = coin.sigma_process_noise^2 ./ max(1 - a.^2, eps);
    end
end

feedbackMean = stateMean + raw.bias;
feedbackVar = stateVar + coin.sigma_sensory_noise^2 + coin.sigma_motor_noise^2;
[mu, v] = validation_mixture_moments(weights, feedbackMean, feedbackVar);
end

function X = normalize_columns(X)
X(~isfinite(X) | X < 0) = 0;
den = sum(X, 1);
bad = den <= 0;
if any(bad)
    X(:, bad) = 1 ./ size(X, 1);
    den(bad) = 1;
end
X = X ./ den;
end
