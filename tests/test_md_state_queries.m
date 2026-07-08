function test_md_state_queries
%TEST_MD_STATE_QUERIES Multi-dimensional grid-based predictive density queries.
%
%   Exercises the MD generalisations of state_probability,
%   state_feedback_probability, state_given_context_probability and
%   predictive_state_feedback_cdf. Builds a 2-D model (correlated process and
%   observation noise so the full-covariance Gaussian-mixture machinery is
%   genuinely used), drives it for a few trials, then checks that each grid
%   density is non-negative and integrates to ~1, that per-context densities
%   are proper, and that the marginal predictive CDF is a valid, monotone
%   distribution function. A scalar-model section confirms the unchanged
%   scalar path still produces normalised densities and a [0,1] CDF.

rng(11);

% ---------------------------------------------------------------------------
% Multi-dimensional model
% ---------------------------------------------------------------------------
N = 2;
a = 0.8;
drift = 0.03;
A = a * eye(N);
d = drift * ones(N, 1);
Q = [1.0e-4, 3.0e-5; 3.0e-5, 1.2e-4];      % correlated process noise
R = [4.0e-4, -1.0e-4; -1.0e-4, 5.0e-4];     % correlated observation noise

coin = RealTimeCOIN('num_particles', 200, 'max_contexts', 2, 'state_dim', N, ...
    'prior_mean_retention', a, 'prior_precision_retention', 1e10, ...
    'prior_mean_drift', drift, 'prior_precision_drift', 1e10, ...
    'process_noise_covariance', Q, 'observation_noise_covariance', R);

% Drive a few trials from the stationary distribution.
Lq = chol(Q, 'lower');
Lr = chol(R, 'lower');
s = d ./ (1 - a);
for t = 1:8
    s = A * s + d + Lq * randn(N, 1);
    y = s + Lr * randn(N, 1);
    coin.observe_q(1);
    coin.observe_y(y);
end

% --- state_probability: non-negative, integrates to ~1 -------------------
[muS, vS] = coin.state_moments();
densState = @(X) coin.state_probability(X);
[intState, valsState] = testutil.integrate2d(densState, muS, sqrt(diag(vS)));
assert(all(valsState >= 0), 'state_probability returned a negative density');
assert(abs(intState - 1) < 0.05, ...
    sprintf('state_probability does not integrate to 1 (got %.4f)', intState));

% Shape: 1-by-K row for an N-by-K grid.
gridPts = muS + [0 0.01 -0.01; 0 -0.01 0.01];
dRow = coin.state_probability(gridPts);
assert(isequal(size(dRow), [1, 3]), 'state_probability output is not 1-by-K');

% --- state_feedback_probability: non-negative, integrates to ~1 ----------
[muF, SigmaF] = coin.predictive_feedback_moments(1);
densFb = @(X) coin.state_feedback_probability(X);
[intFb, valsFb] = testutil.integrate2d(densFb, muF, sqrt(diag(SigmaF)));
assert(all(valsFb >= 0), 'state_feedback_probability returned a negative density');
assert(abs(intFb - 1) < 0.05, ...
    sprintf('state_feedback_probability does not integrate to 1 (got %.4f)', intFb));

% --- state_given_context_probability: Map of proper per-context densities -
ctxDens = coin.state_given_context_probability(gridPts);
assert(isa(ctxDens, 'containers.Map'), 'state_given_context_probability did not return a Map');
ks = ctxDens.keys;
assert(~isempty(ks), 'state_given_context_probability returned no contexts');
for i = 1:numel(ks)
    row = ctxDens(ks{i});
    assert(isequal(size(row), [1, size(gridPts, 2)]), ...
        'per-context density is not 1-by-K');
    assert(all(row >= 0), 'per-context density is negative');
end
% Integrate one context's density to ~1.
ctxFun = @(X) mapValue(coin.state_given_context_probability(X), ks{1});
[intCtx, ~] = testutil.integrate2d(ctxFun, muS, sqrt(diag(vS)));
assert(abs(intCtx - 1) < 0.08, ...
    sprintf('per-context density does not integrate to 1 (got %.4f)', intCtx));

% --- predictive_state_feedback_cdf: N-by-1 vector, valid marginal CDFs ---
coin.observe_q(1);
sigmaF = sqrt(diag(SigmaF));
pMid = coin.predictive_state_feedback_cdf(muF, 1);
assert(isequal(size(pMid), [N, 1]), 'MD predictive CDF is not N-by-1');
assert(all(pMid >= 0 & pMid <= 1), 'MD predictive CDF outside [0,1]');
pLow = coin.predictive_state_feedback_cdf(muF - 8*sigmaF, 1);
pHigh = coin.predictive_state_feedback_cdf(muF + 8*sigmaF, 1);
assert(all(pLow < 1e-3), 'MD predictive CDF does not vanish in the lower tail');
assert(all(pHigh > 1 - 1e-3), 'MD predictive CDF does not approach 1 in the upper tail');
assert(all(pLow <= pMid + 1e-12) && all(pMid <= pHigh + 1e-12), ...
    'MD predictive CDF is not monotone in y');

% ---------------------------------------------------------------------------
% Scalar model: unchanged path still yields a normalised density and [0,1] CDF
% ---------------------------------------------------------------------------
rng(3);
scal = RealTimeCOIN('num_particles', 50, 'max_contexts', 3);
for t = 1:8
    scal.observe_q(1);
    scal.observe_y(0.1 * randn());
end
grid = linspace(-3, 3, 601);
ds = scal.state_probability(grid);
assert(isequal(size(ds), size(grid)), 'scalar state_probability changed output shape');
assert(abs(trapz(grid, ds) - 1) < 0.05, 'scalar state_probability no longer integrates to 1');
df = scal.state_feedback_probability(grid);
assert(abs(trapz(grid, df) - 1) < 0.05, 'scalar state_feedback_probability no longer integrates to 1');
scal.observe_q(1);
pc = scal.predictive_state_feedback_cdf(0.0, 1);
assert(isscalar(pc) && pc >= 0 && pc <= 1, 'scalar predictive CDF not a scalar in [0,1]');
assert(scal.predictive_state_feedback_cdf(-5, 1) < pc + 1e-12 && ...
       pc < scal.predictive_state_feedback_cdf(5, 1) + 1e-12, ...
       'scalar predictive CDF not monotone');

fprintf('test_md_state_queries passed (state integral %.3f, feedback integral %.3f).\n', ...
    intState, intFb);
end

% ---------------------------------------------------------------------------
function v = mapValue(m, key)
%MAPVALUE Fetch a value from a containers.Map (helper for anonymous handles).
v = m(key);
end
