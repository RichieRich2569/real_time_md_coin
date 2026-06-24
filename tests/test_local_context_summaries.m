function test_local_context_summaries
%TEST_LOCAL_CONTEXT_SUMMARIES Fast context summaries avoid global relabelling.

coin = RealTimeCOIN('num_particles', 20, 'max_contexts', 4);
coin.observe_y(0.1);

pred = coin.predicted_context_probabilities_local();
resp = coin.context_responsibilities_local();
count = coin.sampled_context_count_local();

assert(isrow(pred) && numel(pred) == coin.max_contexts + 1, ...
    'Predicted local probabilities should be a Cmax-length row vector');
assert(isrow(resp) && numel(resp) == coin.max_contexts + 1, ...
    'Responsibility local probabilities should be a Cmax-length row vector');
assert(isrow(count) && numel(count) == coin.max_contexts + 1, ...
    'Sampled local counts should be a Cmax-length row vector');
assert(abs(sum(pred) - 1) < 1e-12, 'Predicted local probabilities should normalize');
assert(abs(sum(resp) - 1) < 1e-12, 'Responsibility local probabilities should normalize');
assert(abs(sum(count) - 1) < 1e-12, 'Sampled local counts should normalize');

before = coin.context_alignment();
coin.observe_y(0.2);
coin.predicted_context_probabilities_local();
coin.context_responsibilities_local();
after = coin.context_alignment();
assert(after.cache_state_version ~= before.cache_state_version, ...
    'Local summaries should not refresh the exact alignment cache');
end
