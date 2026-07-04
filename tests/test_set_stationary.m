function test_set_stationary
%TEST_SET_STATIONARY Preserve learned particles while stationaryising state.

rng(17);
test_scalar_stationary_reset();
test_md_stationary_reset();
end

function test_scalar_stationary_reset()
coin = RealTimeCOIN('num_particles', 25, 'max_contexts', 4, 'infer_bias', true);
for t = 1:10
    coin.observe_q(10 + mod(t, 2));
    coin.observe_y(0.03 * t);
end

before = coin.diagnostics();
countMassBefore = sum(before.raw.n_context(:));
assert(countMassBefore > 0, 'Training should create transition counts');

coin.set_stationary();
S = coin.diagnostics();
D = S.raw;
Cmax = coin.max_contexts + 1;

assert(coin.Trial == 0, 'set_stationary should reset the trial counter');
assert(sum(D.n_context(:)) == countMassBefore, ...
    'set_stationary should preserve learned transition counts');

for p = 1:coin.num_particles
    valid = false(1, Cmax);
    valid(1:D.C(p)) = true;
    if D.C(p) < coin.max_contexts
        valid(D.C(p) + 1) = true;
    end
    pi = RealTimeCOIN.stationary_distribution(D.local_transition_matrix(valid, valid, p));
    assert(max(abs(D.predicted_probabilities(valid, p)' - pi)) < 1e-10, ...
        'Predicted probabilities should equal learned stationary context probabilities');
    assert(all(D.predicted_probabilities(~valid, p) == 0), ...
        'Invalid contexts should have zero predicted probability');
end

denomMean = 1 - D.retention;
expectedMean = zeros(size(D.retention));
goodMean = abs(denomMean) > eps;
expectedMean(goodMean) = D.drift(goodMean) ./ denomMean(goodMean);

denomVar = 1 - D.retention.^2;
expectedVar = zeros(size(D.retention));
goodVar = denomVar > eps;
expectedVar(goodVar) = coin.sigma_process_noise^2 ./ denomVar(goodVar);

obsVar = coin.sigma_sensory_noise^2 + coin.sigma_motor_noise^2;
assert(max(abs(D.state_filtered_mean(:) - expectedMean(:))) < 1e-12, ...
    'Scalar filtered means should be stationary under learned dynamics');
assert(max(abs(D.state_filtered_var(:) - expectedVar(:))) < 1e-12, ...
    'Scalar filtered variances should be stationary under learned dynamics');
assert(max(abs(D.state_mean(:) - D.state_filtered_mean(:))) < 1e-12, ...
    'Scalar predictive state means should refresh from stationary filtered means');
assert(max(abs(D.state_feedback_mean(:) - (D.state_mean(:) + D.bias(:)))) < 1e-12, ...
    'Scalar feedback means should refresh after stationary reset');
assert(max(abs(D.state_feedback_var(:) - (D.state_var(:) + obsVar))) < 1e-12, ...
    'Scalar feedback variances should refresh after stationary reset');
end

function test_md_stationary_reset()
coin = RealTimeCOIN('num_particles', 12, 'max_contexts', 3, 'state_dim', 2);
for t = 1:6
    coin.observe_q(100 + mod(t, 2));
    coin.observe_y([0.04 * t; -0.02 * t]);
end

before = coin.diagnostics();
countMassBefore = sum(before.raw.n_context(:));
assert(countMassBefore > 0, 'MD training should create transition counts');

coin.set_stationary();
S = coin.diagnostics();
D = S.raw;
Q = coin.sigma_process_noise^2 * eye(coin.state_dim);

assert(coin.Trial == 0, 'MD set_stationary should reset the trial counter');
assert(sum(D.n_context(:)) == countMassBefore, ...
    'MD set_stationary should preserve learned transition counts');

for p = 1:coin.num_particles
    for c = 1:(coin.max_contexts + 1)
        A = D.Theta(:, 1:coin.state_dim, c, p);
        d = D.Theta(:, coin.state_dim + 1, c, p);
        m = D.state_filtered_mean(:, c, p);
        V = D.state_filtered_cov(:, :, c, p);
        assert(norm(m - (A * m + d), Inf) < 1e-8, ...
            'MD filtered mean should satisfy the stationary mean equation');
        assert(norm(V - (A * V * A' + Q), 'fro') < 1e-8, ...
            'MD filtered covariance should satisfy the stationary covariance equation');
    end
end
end
