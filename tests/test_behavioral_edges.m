function test_behavioral_edges
%TEST_BEHAVIORAL_EDGES Streaming edge cases and diagnostics.

rng(5);

coin = RealTimeCOIN('num_particles', 30, 'max_contexts', 3);
coin.observe_y(0.1);              % observation without cue
coin.observe_q(99);
coin.observe_y(NaN);              % cue-only channel trial
coin.observe_q(42);
coin.observe_y([]);               % second channel style
assert(coin.Trial == 3, 'Trial counter mismatch for edge observations');
assert(abs(sum(coin.responsibilities()) - 1) < 1e-9, 'Responsibilities not normalized');

diag = coin.diagnostics();
assert(all(diag.C <= coin.max_contexts), 'Context cap exceeded');
assert(size(diag.local_transition_matrix, 1) == coin.max_contexts + 1, 'Transition matrix size mismatch');
assert(size(diag.local_cue_matrix, 2) >= 2, 'Streaming cue columns did not expand');

coinBias = RealTimeCOIN('num_particles', 20, 'max_contexts', 2, 'infer_bias', true);
coinBias.observe_q(1);
coinBias.observe_y(0.2);
diagBias = coinBias.diagnostics();
assert(isfield(diagBias, 'bias'), 'Bias diagnostics missing');
assert(all(isfinite(diagBias.bias(:))), 'Bias samples must be finite');

coinDet = RealTimeCOIN('num_particles', 10, 'max_contexts', 1, ...
    'sigma_process_noise', 0, 'sigma_sensory_noise', 0, 'sigma_motor_noise', 0, ...
    'prior_mean_retention', 0.5, 'prior_precision_retention', 1e12, ...
    'prior_mean_drift', 0.1, 'prior_precision_drift', 1e12);
coinDet.observe_y(0.2);
dens = coinDet.state_probability(linspace(-1, 1, 101));
assert(any(isfinite(dens)), 'Deterministic density should not produce all nonfinite values');
end
