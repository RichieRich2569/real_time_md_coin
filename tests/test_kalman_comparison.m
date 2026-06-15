function test_kalman_comparison
%TEST_KALMAN_COMPARISON Compare RealTimeCOIN with a Kalman filter in the single-context limit.

rng(2);

% True parameters
a_true = 0.8;
d_true = 0.05;
q_std = 0.01;
r_std = 0.02;

% Generate synthetic data
T = 8;
s = 0;
y = zeros(1,T);
for t = 1:T
    s = a_true * s + d_true + q_std*randn();
    y(t) = s + r_std*randn();
end

% Run Kalman filter with known parameters
m = d_true/(1 - a_true);
p = q_std^2/(1 - a_true^2);
kf_means = zeros(1,T);
for t = 1:T
    m_pred = a_true*m + d_true;
    p_pred = a_true^2*p + q_std^2;
    K = p_pred/(p_pred + r_std^2);
    m = m_pred + K*(y(t) - m_pred);
    p = (1 - K)*p_pred;
    kf_means(t) = m;
end

% Configure RealTimeCOIN with single context and tight priors
coin = RealTimeCOIN('num_particles', 100, 'max_contexts', 1, ...
    'prior_mean_retention', a_true, 'prior_precision_retention', 1e12, ...
    'prior_mean_drift', d_true, 'prior_precision_drift', 1e12, ...
    'sigma_process_noise', q_std, 'sigma_sensory_noise', r_std, ...
    'sigma_motor_noise', 0);
means_coin = zeros(1,T);

for t = 1:T
    coin.observe_q(1);
    coin.observe_y(y(t));
    grid = linspace(-2,2,401);
    dens = coin.state_probability(grid);
    area = trapz(grid, dens);
    if area > 0
        dens = dens / area;
    end
    mu = trapz(grid, dens .* grid);
    means_coin(t) = mu;
end

% Compare predicted means with Kalman filter means
assert(all(abs(means_coin - kf_means) < 0.15), 'Kalman means differ from COIN predictions beyond tolerance');
end
