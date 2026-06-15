function test_basic
%TEST_BASIC Basic sanity checks for RealTimeCOIN.

rng(1);

% Initialize with few particles
coin = RealTimeCOIN('num_particles', 20, 'max_contexts', 3);
% Initially, all particles in context 1
probs = coin.context_probabilities();
assert(numel(probs.keys) == 1, 'Initial context count mismatch');
assert(abs(probs(1) - 1.0) < 1e-12, 'Initial context probability not 1');

% Perform a single observation
coin.observe_q(1);
coin.observe_y(0.2);
probs = coin.context_probabilities();
total = 0;
ks = probs.keys;
for i = 1:numel(ks)
    total = total + probs(ks{i});
end
assert(abs(total - 1) < 1e-6, 'Context probabilities do not sum to 1');

% Check state probability integrates to something reasonable
grid = linspace(-3,3,601);
dens = coin.state_probability(grid);
intVal = trapz(grid, dens);
assert(intVal > 0 && intVal < 2, 'State probability integral out of bounds');

diag = coin.diagnostics();
assert(isfield(diag, 'predicted_probabilities'), 'Diagnostics missing predicted probabilities');
assert(isfield(diag, 'alignment'), 'Diagnostics missing global alignment');
assert(size(diag.predicted_probabilities, 2) == sum(diag.alignment.modal_particle_mask), ...
    'Diagnostics should expose the aligned modal particle subset');
assert(abs(sum(coin.predicted_context_probabilities()) - 1) < 1e-9, 'Predicted probabilities do not sum to 1');
assert(isfinite(coin.motor_output()), 'Motor output must be finite');
end
