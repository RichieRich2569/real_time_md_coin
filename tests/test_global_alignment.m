function test_global_alignment
%TEST_GLOBAL_ALIGNMENT Lazy global context alignment for context-facing APIs.

rng(6);

coin = testutil.alignmentFixture(false);
alignment = coin.context_alignment();

assert(alignment.K == 2, 'Alignment should use the modal cardinality');
assert(all(alignment.modal_particle_mask), 'All fixture particles should be modal');
assert(isequal(alignment.assignment(1:2,1), [1;2]), 'Anchor particle should keep canonical labels');
assert(isequal(alignment.assignment(1:2,2), [2;1]), 'Swapped particle should map back to global labels');
assert(abs(alignment.global_contexts.state_mean(1) + 1) < 1e-6, 'Global context 1 prototype mismatch');
assert(abs(alignment.global_contexts.state_mean(2) - 1) < 1e-6, 'Global context 2 prototype mismatch');

resp = coin.context_responsibilities();
assert(abs(resp(1) - 0.6) < 1e-12, 'Global responsibility for context 1 mismatch');
assert(abs(resp(2) - 0.3) < 1e-12, 'Global responsibility for context 2 mismatch');
assert(abs(resp(3) - 0.1) < 1e-12, 'Novel responsibility bucket mismatch');

pred = coin.predicted_context_probabilities();
assert(max(abs(pred(1:3) - [0.7 0.2 0.1])) < 1e-12, 'Global predicted probabilities mismatch');

coinModal = testutil.alignmentFixture(true);
alignmentModal = coinModal.context_alignment();
assert(alignmentModal.K == 2, 'Modal subset should keep the two-context cardinality');
assert(sum(alignmentModal.modal_particle_mask) == 3, 'Exactly three particles should be in the modal subset');
respModal = coinModal.context_responsibilities();
assert(abs(respModal(1) - 0.6) < 1e-12, 'Non-modal particles should not dilute modal summaries');
diagModal = coinModal.diagnostics();
assert(size(diagModal.responsibilities, 2) == 3, 'Diagnostics should expose modal particles only');
assert(size(diagModal.raw.responsibilities, 2) == 4, 'Raw diagnostics should preserve all particles');

coinLazy = RealTimeCOIN('num_particles', 12, 'max_contexts', 3);
coinLazy.observe_y(0.1);
a1 = coinLazy.context_alignment();
a2 = coinLazy.context_alignment();
assert(a1.cache_state_version == a2.cache_state_version, 'Repeated alignment calls should reuse the cached state version');
coinLazy.observe_y(0.2);
a3 = coinLazy.context_alignment();
assert(a3.cache_state_version ~= a1.cache_state_version, 'A new observation should invalidate the alignment cache');

coinWarm = testutil.alignmentFixture(false);
w1 = coinWarm.context_alignment();
diagWarm = coinWarm.diagnostics();
rawWarm = diagWarm.raw;
warmTemplate = RealTimeCOIN('num_particles', 3, 'max_contexts', 3);
coinWarmSeeded = testutil.loadFixtureModel(warmTemplate, rawWarm, w1);
w2 = coinWarmSeeded.context_alignment();
assert(w2.used_seed, 'Post-update alignment should warm-start from the previous seed');
assert(isequal(w1.assignment(1:2,1:3), w2.assignment(1:2,1:3)), ...
    'Warm-started alignment should preserve stable label order');

coinTen = testutil.largeAssignmentFixture();
ten = coinTen.context_alignment();
assert(ten.K == 10, 'Large fixture should align ten contexts');
expected = 10:-1:1;
assert(isequal(ten.assignment(1:10,2)', expected), ...
    'Ten-context assignment did not recover the known reverse permutation');
end
