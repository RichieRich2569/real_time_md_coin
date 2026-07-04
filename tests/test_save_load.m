function test_save_load
%TEST_SAVE_LOAD Ensure that saving and loading preserves or resets state.

rng(3);

coin = RealTimeCOIN('num_particles', 20, 'max_contexts', 3, 'infer_bias', true);

% Feed some observations
coin.observe_q(10);
coin.observe_y(0.3);
coin.observe_q(20);
coin.observe_y(0.1);
before = coin.diagnostics();
countMassBefore = sum(before.raw.n_context(:));

% Save with setStationary = true
tmpfile = [tempname, '.mat'];
coin.saveModel(tmpfile, true);

% Load into a new object
coin2 = RealTimeCOIN('num_particles', 1); % dummy initialisation
coin2.loadModel(tmpfile);

assert(coin2.Trial == 0, 'Trial count should reset to zero after stationary save');
after = coin2.diagnostics();
assert(sum(after.raw.n_context(:)) == countMassBefore, ...
    'Stationary save should preserve learned transition counts');
assert(size(after.raw.n_cue, 2) == size(before.raw.n_cue, 2), ...
    'Stationary save should preserve learned cue columns');
savedFile = load(tmpfile);
assert(isequal(savedFile.model.cue_values, [10 20]), ...
    'Stationary save should preserve raw cue-value mapping');

% Save without reset
coin.observe_q(1);
coin.observe_y(0.1);
coin.saveModel(tmpfile, false);
coin3 = RealTimeCOIN('num_particles', 1);
coin3.loadModel(tmpfile);
assert(coin3.Trial == coin.Trial, 'Trial count mismatch after save/load');
probs_old = coin.context_responsibilities();
probs_new = coin3.context_responsibilities();
keys_old = probs_old.keys;
keys_new = probs_new.keys;
assert(numel(keys_old) == numel(keys_new), 'Context key count mismatch');
for i = 1:numel(keys_old)
    k = keys_old{i};
    assert(abs(probs_old(k) - probs_new(k)) < 1e-6, 'Context probability mismatch after save/load');
end
end
