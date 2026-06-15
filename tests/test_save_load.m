function test_save_load
%TEST_SAVE_LOAD Ensure that saving and loading preserves or resets state.

rng(3);

coin = RealTimeCOIN('num_particles', 20, 'max_contexts', 3, 'infer_bias', true);

% Feed some observations
coin.observe_q(1);
coin.observe_y(0.3);

% Save with setStationary = true
tmpfile = [tempname, '.mat'];
coin.saveModel(tmpfile, true);

% Load into a new object
coin2 = RealTimeCOIN('num_particles', 1); % dummy initialisation
coin2.loadModel(tmpfile);

assert(coin2.Trial == 0, 'Trial count should reset to zero after stationary save');
probs = coin2.context_probabilities();
keys = probs.keys;
assert(numel(keys) == 1 && keys{1} == 1, 'After reset only context 1 should be present');

% Save without reset
coin.observe_q(1);
coin.observe_y(0.1);
coin.saveModel(tmpfile, false);
coin3 = RealTimeCOIN('num_particles', 1);
coin3.loadModel(tmpfile);
assert(coin3.Trial == coin.Trial, 'Trial count mismatch after save/load');
probs_old = coin.context_probabilities();
probs_new = coin3.context_probabilities();
keys_old = probs_old.keys;
keys_new = probs_new.keys;
assert(numel(keys_old) == numel(keys_new), 'Context key count mismatch');
for i = 1:numel(keys_old)
    k = keys_old{i};
    assert(abs(probs_old(k) - probs_new(k)) < 1e-6, 'Context probability mismatch after save/load');
end
end
