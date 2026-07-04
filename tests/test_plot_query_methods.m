function test_plot_query_methods
%TEST_PLOT_QUERY_METHODS Invariants for the COIN-plottable query methods.
%
%   Exercises the query methods added so RealTimeCOIN can reproduce every
%   quantity COIN.m is able to plot (c*/component scalars, per-context parameter
%   densities, and transition/cue/stationary distributions). Rather than a noisy
%   cross-model comparison, it asserts deterministic invariants that pin down the
%   formulas: probability vectors normalise, the stationary distribution is a
%   fixed point of the transition matrix, each per-context density peaks at that
%   context's aligned prototype moment, and the explicit/implicit identities and
%   multi-dimensional guards hold.

rng(7);
tol = 1e-9;

% ---------------------------------------------------------------- scalar model
m = RealTimeCOIN('num_particles', 60, 'max_contexts', 5, 'infer_bias', true);
perturb = [zeros(1,8), 0.3*ones(1,12), -0.3*ones(1,12)];
cues    = [ones(1,8), 2*ones(1,12), 3*ones(1,12)];
for t = 1:numel(perturb)
    m.observe_q(cues(t));
    m.observe_y(perturb(t));
end

alignment = m.context_alignment();
K = alignment.K;
gc = alignment.global_contexts;

% --- Group 1: component identities and bounds ---
assertEqualScalar('explicit == state_cstar1', m.explicit_component(), m.state_cstar1(), tol);
[mu, ~] = m.state_moments();
assertEqualScalar('implicit == motor_output - avg_state', ...
    m.implicit_component(), m.motor_output() - mu, tol);
assertInRange('predicted_probability_cstar1', m.predicted_probability_cstar1(), 0, 1);
assertInRange('predicted_probability_cstar3', m.predicted_probability_cstar3(), 0, 1);
assertInRange('kalman_gain_cstar1', m.kalman_gain_cstar1(), 0, 1);
assertInRange('kalman_gain_cstar2', m.kalman_gain_cstar2(), 0, 1);

% --- Group 2: per-context densities peak at the aligned prototype mean ---
rgrid = linspace(0.85, 1.0, 4001);
dgrid = linspace(-0.05, 0.05, 4001);
bgrid = linspace(-0.6, 0.6, 8001);
assertDensityPeaks('retention', m.retention_given_context_probability(rgrid), rgrid, gc.dynamics_mean(1,:));
assertDensityPeaks('drift', m.drift_given_context_probability(dgrid), dgrid, gc.dynamics_mean(2,:));
assertDensityPeaks('bias|context', m.bias_given_context_probability(bgrid), bgrid, gc.bias_mean);

% marginal bias density: nonnegative, finite, integrates to ~1 on a wide grid
bp = m.bias_probability(bgrid);
assertTrue('bias_probability nonneg/finite', all(bp >= 0) && all(isfinite(bp)));
assertEqualScalar('bias_probability integrates to 1', trapz(bgrid, bp), 1, 5e-3);

% --- Group 3: transition/cue/stationary normalisation and fixed point ---
ltp = m.local_transition_probabilities();
assertSize('local_transition_probabilities', ltp, [K, K+1]);
assertTrue('local transition rows sum to 1', max(abs(sum(ltp,2) - 1)) < 1e-9);

lcp = m.local_cue_probabilities();
assertSize('local_cue_probabilities rows', lcp, [K, size(lcp,2)]);
assertTrue('local cue rows sum to 1', max(abs(sum(lcp,2) - 1)) < 1e-9);

sp = m.stationary_context_probabilities();
assertSize('stationary_context_probabilities', sp, [1, K]);
assertEqualScalar('stationary sums to 1', sum(sp), 1, 1e-9);
% fixed-point property: sp * Tblock == sp (Tblock = transition over known contexts)
Tblock = ltp(:, 1:K);
Tblock = Tblock ./ sum(Tblock, 2);
assertTrue('stationary is a fixed point of T', max(abs(sp * Tblock - sp)) < 1e-6);

gt = m.global_transition_probabilities();
assertEqualScalar('global_transition sums to 1', sum(gt), 1, 1e-9);
gcue = m.global_cue_probabilities();
assertEqualScalar('global_cue sums to 1', sum(gcue), 1, 1e-9);

% --- cue methods require cues ---
mNoCue = RealTimeCOIN('num_particles', 20, 'max_contexts', 3);
mNoCue.observe_y(0.1); mNoCue.observe_y(0.2);
mustError('local_cue_probabilities without cues', @() mNoCue.local_cue_probabilities(), 'RealTimeCOIN:NoCues');
mustError('global_cue_probabilities without cues', @() mNoCue.global_cue_probabilities(), 'RealTimeCOIN:NoCues');

% bias methods require infer_bias
mNoBias = RealTimeCOIN('num_particles', 20, 'max_contexts', 3);
mNoBias.observe_y(0.1); mNoBias.observe_y(0.2);
mustError('bias_given_context requires infer_bias', ...
    @() mNoBias.bias_given_context_probability(bgrid), 'RealTimeCOIN:BiasNotInferred');
mustError('bias_probability requires infer_bias', ...
    @() mNoBias.bias_probability(bgrid), 'RealTimeCOIN:BiasNotInferred');

% ------------------------------------------------------------ multi-dim model
rng(8);
m2 = RealTimeCOIN('num_particles', 40, 'max_contexts', 4, 'state_dim', 2);
for t = 1:25
    m2.observe_y(0.2*randn(2,1));
end
assertSize('MD explicit_component', m2.explicit_component(), [2, 1]);
assertSize('MD implicit_component', m2.implicit_component(), [2, 1]);
assertSize('MD state_cstar2', m2.state_cstar2(), [2, 1]);
K2 = m2.context_alignment().K;
sp2 = m2.stationary_context_probabilities();
assertEqualScalar('MD stationary sums to 1', sum(sp2), 1, 1e-9);
assertSize('MD local_transition_probabilities', m2.local_transition_probabilities(), [K2, K2+1]);

% scalar-only quantities must reject the MD model
mustError('MD kalman_gain_cstar1', @() m2.kalman_gain_cstar1(), 'RealTimeCOIN:ScalarModelOnly');
mustError('MD kalman_gain_cstar2', @() m2.kalman_gain_cstar2(), 'RealTimeCOIN:ScalarModelOnly');
mustError('MD retention_given_context', @() m2.retention_given_context_probability(rgrid), 'RealTimeCOIN:ScalarModelOnly');
mustError('MD drift_given_context', @() m2.drift_given_context_probability(dgrid), 'RealTimeCOIN:ScalarModelOnly');
mustError('MD bias_given_context', @() m2.bias_given_context_probability(bgrid), 'RealTimeCOIN:ScalarModelOnly');

fprintf('test_plot_query_methods passed (scalar K=%d, MD K=%d).\n', K, K2);
end

% ------------------------------------------------------------------- helpers
function assertTrue(name, cond)
    if ~cond
        error('test_plot_query_methods:failed', 'FAILED: %s', name);
    end
end

function assertEqualScalar(name, a, b, tol)
    if any(abs(a(:) - b(:)) > tol)
        error('test_plot_query_methods:failed', 'FAILED: %s (max diff %.3g)', name, max(abs(a(:)-b(:))));
    end
end

function assertInRange(name, v, lo, hi)
    if any(v(:) < lo - 1e-12) || any(v(:) > hi + 1e-12)
        error('test_plot_query_methods:failed', 'FAILED: %s = %.6g not in [%g,%g]', name, v, lo, hi);
    end
end

function assertSize(name, v, expected)
    if ~isequal(size(v), expected)
        error('test_plot_query_methods:failed', 'FAILED: %s size [%s] != [%s]', ...
            name, num2str(size(v)), num2str(expected));
    end
end

function assertDensityPeaks(name, dmap, grid, means)
    % Each context's density must peak at the grid point nearest its prototype mean.
    ks = dmap.keys;
    dx = grid(2) - grid(1);
    for i = 1:numel(ks)
        c = ks{i};
        d = dmap(c);
        [~, iPeak] = max(d);
        if abs(grid(iPeak) - means(c)) > 2*dx
            error('test_plot_query_methods:failed', ...
                'FAILED: %s context %d peak at %.5g, expected mean %.5g', ...
                name, c, grid(iPeak), means(c));
        end
    end
end

function mustError(name, fn, expectedId)
    try
        fn();
        error('test_plot_query_methods:failed', 'FAILED: %s did not error', name);
    catch e
        if ~strcmp(e.identifier, expectedId)
            rethrow(e);
        end
    end
end
