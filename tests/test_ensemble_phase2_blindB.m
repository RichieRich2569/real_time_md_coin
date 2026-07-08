function test_ensemble_phase2_blindB
%TEST_ENSEMBLE_PHASE2 Blind behavioural tests for RealTimeCOINEnsemble Phase 2.
%
%   Authored strictly against docs/SPEC_ensemble.md Part 10 (Phase 2), NOT
%   against any implementation. In this worktree the six Phase-2 methods are
%   NaN/empty stubs, so these tests are EXPECTED to fail here; they are a genuine
%   independent oracle for the real implementation. Plain-function style to match
%   the rest of tests/ (see run_tests.m, test_ensemble_blindB.m).
%
%   Phase 2 adds cross-run context alignment onto a common reference frame (the
%   member with the most contexts) and two averaging rules:
%     * zero-fill / divide-by-R for probability vectors
%       (responsibilities_vector, predicted_context_probabilities_vector,
%        sampled_context_count, stationary_context_probabilities) so they sum 1;
%     * NaN-omit mean for per-context densities
%       (state_given_context_probability, state_feedback_given_context_probability).
%
%   Coverage (mapped to SPEC 10.5) -- deliberately chosen so the aligner need NOT
%   be re-implemented:
%     1. runs==1 reduction (scalar + MD): each Phase-2 query equals its single
%        member's query exactly (identity matching, Kref = K_1).
%     2. Probability conservation: the four probability quantities sum to 1, are
%        nonnegative, and have the right length.
%     3. Member-permutation invariance (via order-independent naive oracle in the
%        trivial regime + reproducibility; see the documented limitation below).
%     4. Reproducibility & executor invariance (max_cores / segment_length).
%     5. Trivial-alignment regime: a single shared context, so the aligned
%        average equals the naive slot-by-slot / key-by-key member average.
%     6. Density normalisation: each per-context density integrates to ~1 (trapz).
%     7. Shape / key correctness.
%
%   Spec ambiguities resolved here (documented for the reviewer):
%     * SPEC 3.1 permits "Threefry OR Philox" as the sub-streamable generator.
%       The exact-reproduction oracle for the runs==1 checks must know which one;
%       we DETECT it at runtime from the live Phase-1 motor_output (which is not
%       stubbed) and use the matching generator. If neither matches, the RNG
%       contract is unmet and the affected checks fail by assertion.
%     * "sums to 1 within a few eps" (10.5.2): implemented as abs(sum-1) <= 1e-12,
%       comfortably above the O(R*eps) accumulation of a zero-fill/divide-by-R
%       average yet far tighter than any modelling tolerance.
%     * Trivial-alignment regime: enforced as a SINGLE shared context (K==1 for
%       every member). With one instantiated context the global labels are
%       unambiguous AND the novel slot sits at the same position (slot 2) in every
%       member, so the naive slot-by-slot / key-by-key member average provably
%       equals the reference-frame aligned average. The check asserts K==1 for all
%       members (a constant perturbation makes this robust); if a member splits,
%       the assertion flags the scenario rather than silently passing.
%     * Member-permutation invariance (10.5.3): the ensemble exposes no way to
%       reorder its internal members, so we establish the property two ways -- (a)
%       reproducibility (same seed => bit-identical) and (b) in the trivial regime
%       we show our naive equal-weight member average is invariant to member order
%       AND that the ensemble equals that average, hence the ensemble is
%       order-invariant. Full internal-reorder coverage is out of reach by design.

close_all_safe();

check_runs_one_reduction_scalar();
check_runs_one_reduction_md();
check_probability_conservation();
check_reproducibility();
check_executor_invariance();
check_trivial_alignment_and_permutation();
check_density_normalisation();

fprintf('test_ensemble_phase2: all Phase-2 contract checks passed.\n');
end

% ========================================================================= %
% Guarantee 1 (scalar): runs==1 reduces exactly to the single seeded member.
% ========================================================================= %
function check_runs_one_reduction_scalar()
mp = {'num_particles', 40, 'max_contexts', 4};
seed = 1010;
qCell = {1, 2, 1, NaN, 2};
hasQ  = [true, true, true, true, true];
yCell = {0.2, -0.1, 0.05, NaN, 0.3};
grid = linspace(-1, 1, 17);        % 1xK scalar grid

gen = detect_generator(seed, mp, {1, 2}, {0.1, -0.05});
testutil.assertTrue('runs==1 scalar: RNG generator detected', ~isempty(gen));

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, mp{:});
members = build_members(gen, 1, seed, mp);
saved = RandStream.getGlobalStream();
restore = onCleanup(@() RandStream.setGlobalStream(saved));

for t = 1:numel(yCell)
    if hasQ(t)
        ens.observe_q(qCell{t});
    end
    ens.observe_y(yCell{t});

    RandStream.setGlobalStream(members.streams{1});
    if hasQ(t)
        members.obj{1}.observe_q(qCell{t});
    end
    members.obj{1}.observe_y(yCell{t});
    RandStream.setGlobalStream(saved);

    single = members.obj{1};
    tag = sprintf('runs==1 scalar (t=%d)', t);

    assert_isequaln([tag ' responsibilities_vector'], ...
        ens.responsibilities_vector(), single.responsibilities_vector());
    assert_isequaln([tag ' predicted_context_probabilities_vector'], ...
        ens.predicted_context_probabilities_vector(), ...
        single.predicted_context_probabilities_vector());
    assert_isequaln([tag ' sampled_context_count'], ...
        ens.sampled_context_count(), single.sampled_context_count());
    assert_isequaln([tag ' stationary_context_probabilities'], ...
        ens.stationary_context_probabilities(), ...
        single.stationary_context_probabilities());
    assert_map_equal([tag ' state_given_context_probability'], ...
        ens.state_given_context_probability(grid), ...
        single.state_given_context_probability(grid));
    assert_map_equal([tag ' state_feedback_given_context_probability'], ...
        ens.state_feedback_given_context_probability(grid), ...
        single.state_feedback_given_context_probability(grid));
end
RandStream.setGlobalStream(saved);
end

% ========================================================================= %
% Guarantee 1 (multi-dimensional, state_dim==2): runs==1 reduction.
% ========================================================================= %
function check_runs_one_reduction_md()
mp = {'num_particles', 40, 'max_contexts', 4, 'state_dim', 2};
seed = 2020;
qCell = {1, 1, 2, 2, 1};
hasQ  = [true, true, true, true, true];
yCell = {[0.10; 0.05], [0.20; -0.10], [NaN; 0.10], [0.15; 0.20], []};
grid = [linspace(-0.4, 0.4, 7); linspace(-0.3, 0.3, 7)];     % 2xK MD grid

gen = detect_generator(seed, mp, {1, 2}, {[0.1; 0.0], [-0.05; 0.1]});
testutil.assertTrue('runs==1 MD: RNG generator detected', ~isempty(gen));

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, mp{:});
members = build_members(gen, 1, seed, mp);
saved = RandStream.getGlobalStream();
restore = onCleanup(@() RandStream.setGlobalStream(saved));

for t = 1:numel(yCell)
    if hasQ(t)
        ens.observe_q(qCell{t});
    end
    ens.observe_y(yCell{t});

    RandStream.setGlobalStream(members.streams{1});
    if hasQ(t)
        members.obj{1}.observe_q(qCell{t});
    end
    members.obj{1}.observe_y(yCell{t});
    RandStream.setGlobalStream(saved);

    single = members.obj{1};
    tag = sprintf('runs==1 MD (t=%d)', t);

    assert_isequaln([tag ' responsibilities_vector'], ...
        ens.responsibilities_vector(), single.responsibilities_vector());
    assert_isequaln([tag ' predicted_context_probabilities_vector'], ...
        ens.predicted_context_probabilities_vector(), ...
        single.predicted_context_probabilities_vector());
    assert_isequaln([tag ' sampled_context_count'], ...
        ens.sampled_context_count(), single.sampled_context_count());
    assert_isequaln([tag ' stationary_context_probabilities'], ...
        ens.stationary_context_probabilities(), ...
        single.stationary_context_probabilities());
    assert_map_equal([tag ' state_given_context_probability'], ...
        ens.state_given_context_probability(grid), ...
        single.state_given_context_probability(grid));
    assert_map_equal([tag ' state_feedback_given_context_probability'], ...
        ens.state_feedback_given_context_probability(grid), ...
        single.state_feedback_given_context_probability(grid));
end
RandStream.setGlobalStream(saved);
end

% ========================================================================= %
% Guarantee 2 & 7: probability conservation + shape/key correctness (R>1).
% ========================================================================= %
function check_probability_conservation()
mp = {'num_particles', 60, 'max_contexts', 4};
R = 4;
seed = 3030;
mc = 4;                                    % max_contexts
% A two-phase perturbation (positive then negative) instantiates >= 1 context.
qCell = {1, 1, 2, 2, 1, 2, 1};
hasQ  = true(1, numel(qCell));
yCell = {0.30, 0.35, -0.30, -0.35, 0.32, -0.28, 0.30};
grid = linspace(-2, 2, 41);
sumTol = 1e-12;

gen = detect_generator(seed, mp, {1, 2}, {0.1, -0.05});
testutil.assertTrue('conservation: RNG generator detected', ~isempty(gen));

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
drive_ens(ens, qCell, yCell, hasQ);

% Independent oracle members (same substream contract) to learn Kref and the
% per-member context counts for the shape assertions (single-member API only).
members = build_members(gen, R, seed, mp);
drive_members(members, qCell, yCell, hasQ);
Ks = member_context_counts(members);
Kref = max(Ks);
testutil.assertTrue('conservation: at least one context instantiated (Kref>=1)', ...
    Kref >= 1);

% -- the three (max_contexts+1) probability vectors --
vecMethods = {'responsibilities_vector', ...
              'predicted_context_probabilities_vector', ...
              'sampled_context_count'};
for i = 1:numel(vecMethods)
    v = ens.(vecMethods{i})();
    name = vecMethods{i};
    testutil.assertSize(['conservation ' name ' length'], v, [1, mc + 1]);
    testutil.assertTrue(['conservation ' name ' finite'], all(isfinite(v)));
    testutil.assertTrue(['conservation ' name ' nonnegative'], all(v >= -1e-12));
    testutil.assertClose(sum(v), 1, sumTol, ['conservation ' name ' sums to 1']);
end

% -- stationary_context_probabilities: 1xKref, nonneg, sums to 1 --
sp = ens.stationary_context_probabilities();
testutil.assertSize('conservation stationary length', sp, [1, Kref]);
testutil.assertTrue('conservation stationary finite', all(isfinite(sp)));
testutil.assertTrue('conservation stationary nonnegative', all(sp >= -1e-12));
testutil.assertClose(sum(sp), 1, sumTol, 'conservation stationary sums to 1');

% -- density Map key/shape correctness (keys subset of 1..Kref) --
m1 = ens.state_given_context_probability(grid);
m2 = ens.state_feedback_given_context_probability(grid);
assert_density_map_shape('conservation state_given_context_probability', ...
    m1, Kref, numel(grid));
assert_density_map_shape('conservation state_feedback_given_context_probability', ...
    m2, Kref, numel(grid));
end

% ========================================================================= %
% Guarantee 4a: reproducibility -- same (seed, runs) => bit-identical Phase-2.
% ========================================================================= %
function check_reproducibility()
mp = {'num_particles', 40, 'max_contexts', 4};
R = 4;
seed = 4040;
qCell = {1, 2, 1, NaN, 2, 1};
hasQ  = [true, true, true, true, true, true];
yCell = {0.2, -0.1, 0.3, 0.05, -0.2, 0.15};
grid = linspace(-1, 1, 15);

a = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
b = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
for t = 1:numel(yCell)
    if hasQ(t)
        a.observe_q(qCell{t}); b.observe_q(qCell{t});
    end
    a.observe_y(yCell{t}); b.observe_y(yCell{t});
    assert_phase2_bit_equal(sprintf('reproducibility (t=%d)', t), a, b, grid);
end
end

% ========================================================================= %
% Guarantee 4b: executor invariance -- serial vs parallel and different
% segment_length must give bit-identical Phase-2 outputs (read off the live
% stepping state, since simulate() traces do not carry context-indexed queries).
% ========================================================================= %
function check_executor_invariance()
mp = {'num_particles', 40, 'max_contexts', 4};
R = 4;
seed = 5050;
qCell = {1, 2, 2, 1, NaN, 2};
hasQ  = [true, true, true, true, true, true];
yCell = {0.15, -0.05, 0.25, 0.2, 0.1, -0.15};
grid = linspace(-1, 1, 15);

serial   = RealTimeCOINEnsemble('runs', R, 'seed', seed, 'max_cores', 0, ...
    'segment_length', 1, mp{:});
parallel = RealTimeCOINEnsemble('runs', R, 'seed', seed, 'max_cores', 2, ...
    'segment_length', 3, mp{:});
for t = 1:numel(yCell)
    if hasQ(t)
        serial.observe_q(qCell{t}); parallel.observe_q(qCell{t});
    end
    serial.observe_y(yCell{t}); parallel.observe_y(yCell{t});
    assert_phase2_bit_equal(sprintf('executor invariance (t=%d)', t), ...
        serial, parallel, grid);
end
end

% ========================================================================= %
% Guarantee 5 & 3: trivial-alignment regime + member-permutation invariance.
%   A constant perturbation instantiates exactly ONE shared context per member,
%   so the reference-frame aligned average equals the naive slot-by-slot
%   (vectors) / key-by-key (densities) member average. Because that naive
%   equal-weight average is invariant to member order (and equals the ensemble),
%   the ensemble is member-permutation invariant.
% ========================================================================= %
function check_trivial_alignment_and_permutation()
mp = {'num_particles', 80, 'max_contexts', 4};
R = 4;
seed = 6060;
mc = 4;
nTrials = 5;
% Constant cue + constant, consistent perturbation => a single context, robustly.
qCell = num2cell(ones(1, nTrials));
hasQ  = true(1, nTrials);
yCell = num2cell(0.3 * ones(1, nTrials));
grid = linspace(-2, 2, 41);
tol = 1e-9;

gen = detect_generator(seed, mp, {1, 1}, {0.3, 0.3});
testutil.assertTrue('trivial: RNG generator detected', ~isempty(gen));

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
drive_ens(ens, qCell, yCell, hasQ);

members = build_members(gen, R, seed, mp);
drive_members(members, qCell, yCell, hasQ);
Ks = member_context_counts(members);
% Trivial regime precondition: exactly one shared context in every member.
testutil.assertTrue('trivial: every member has exactly one context', all(Ks == 1));
Kref = 1;

% -- probability vectors: ensemble == plain member mean (zero-fill == raw) --
vecMethods = {'responsibilities_vector', ...
              'predicted_context_probabilities_vector', ...
              'sampled_context_count'};
for i = 1:numel(vecMethods)
    name = vecMethods{i};
    vEns = ens.(name)();
    vNaive = naive_vector_mean(members.obj, name);
    testutil.assertSize(['trivial ' name ' length'], vEns, [1, mc + 1]);
    testutil.assertClose(vEns, vNaive, tol, ['trivial ' name ' == naive member mean']);

    % Permutation invariance of the equal-weight average: reversing member order
    % must not change the mean (up to fp reordering).
    vFlip = naive_vector_mean(members.obj(end:-1:1), name);
    testutil.assertClose(vNaive, vFlip, tol, ...
        ['permutation ' name ' invariant to member order']);
end

% -- stationary: single context => [1]; ensemble matches --
spEns = ens.stationary_context_probabilities();
testutil.assertSize('trivial stationary length', spEns, [1, Kref]);
testutil.assertClose(spEns, 1, tol, 'trivial stationary == [1]');

% -- per-context densities: ensemble Map == naive key-by-key member mean --
assert_trivial_density('trivial state_given_context_probability', ...
    ens, members.obj, 'state_given_context_probability', grid, tol);
assert_trivial_density('trivial state_feedback_given_context_probability', ...
    ens, members.obj, 'state_feedback_given_context_probability', grid, tol);
end

% ========================================================================= %
% Guarantee 6: per-context densities integrate to ~1 over a wide, fine grid.
% ========================================================================= %
function check_density_normalisation()
mp = {'num_particles', 80, 'max_contexts', 4};
R = 3;
seed = 7070;
qCell = {1, 1, 2, 2, 1, 2};
hasQ  = true(1, numel(qCell));
yCell = {0.30, 0.35, -0.30, -0.35, 0.32, -0.28};
grid = linspace(-3, 3, 601);       % wide & fine for trapz
intTol = 0.05;                     % discretisation tolerance (cf. test_md_state_queries)

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
drive_ens(ens, qCell, yCell, hasQ);

dens = ens.state_given_context_probability(grid);
ks = dens.keys;
testutil.assertTrue('density normalisation: at least one context density', ...
    ~isempty(ks));
for i = 1:numel(ks)
    row = dens(ks{i});
    testutil.assertSize(sprintf('density row shape (key=%g)', ks{i}), ...
        row, [1, numel(grid)]);
    testutil.assertTrue(sprintf('density nonnegative (key=%g)', ks{i}), ...
        all(row >= -1e-12));
    integ = trapz(grid, row);
    testutil.assertClose(integ, 1, intTol, ...
        sprintf('per-context density integrates to ~1 (key=%g)', ks{i}));
end
end

% ========================================================================= %
% ---------------------------- oracle helpers ----------------------------- %
% ========================================================================= %

function gen = detect_generator(seed, memberParams, qFirst, yFirst)
%DETECT_GENERATOR Identify which sub-streamable generator the ensemble uses.
%   Drives a runs==1 ensemble over a short (q,y) prefix and compares its Phase-1
%   motor_output (not stubbed) against a canonical substream-1 RealTimeCOIN built
%   with each candidate generator. Returns the matching generator name, or '' if
%   none matches (RNG contract SPEC 3 unmet).
cands = {'Threefry', 'Philox'};
gen = '';

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, memberParams{:});
for t = 1:numel(yFirst)
    ens.observe_q(qFirst{t});
    ens.observe_y(yFirst{t});
end
target = ens.motor_output();
if any(~isfinite(target(:)))
    return;   % degenerate -- cannot detect
end

saved = RandStream.getGlobalStream();
for gi = 1:numel(cands)
    st = RandStream.create(cands{gi}, 'NumStreams', 1, 'StreamIndices', 1, ...
        'Seed', seed);
    RandStream.setGlobalStream(st);
    m = RealTimeCOIN(memberParams{:});
    for t = 1:numel(yFirst)
        m.observe_q(qFirst{t});
        m.observe_y(yFirst{t});
    end
    cand = m.motor_output();
    RandStream.setGlobalStream(saved);
    if isequal(size(cand), size(target)) && ...
            all(abs(cand(:) - target(:)) <= 1e-9 + 1e-9 * abs(target(:)))
        gen = cands{gi};
        return;
    end
end
RandStream.setGlobalStream(saved);
end

function members = build_members(gen, runs, seed, memberParams)
%BUILD_MEMBERS Construct R RealTimeCOIN members under per-run substreams.
%   member k is built under substream k of the (gen, NumStreams=runs, Seed=seed)
%   family (SPEC 3.1) -- matching @RealTimeCOINEnsemble/private/makeMemberStream.
saved = RandStream.getGlobalStream();
members = struct('obj', {cell(1, runs)}, 'streams', {cell(1, runs)});
for k = 1:runs
    members.streams{k} = RandStream.create(gen, 'NumStreams', runs, ...
        'StreamIndices', k, 'Seed', seed);
    RandStream.setGlobalStream(members.streams{k});
    members.obj{k} = RealTimeCOIN(memberParams{:});
end
RandStream.setGlobalStream(saved);
end

function drive_members(members, qCell, yCell, hasQ)
%DRIVE_MEMBERS Step each oracle member under its own substream (queries are
%   read-only, so they draw no randomness and are read with the global stream
%   restored).
saved = RandStream.getGlobalStream();
R = numel(members.obj);
for t = 1:numel(yCell)
    for k = 1:R
        RandStream.setGlobalStream(members.streams{k});
        if hasQ(t)
            members.obj{k}.observe_q(qCell{t});
        end
        members.obj{k}.observe_y(yCell{t});
    end
end
RandStream.setGlobalStream(saved);
end

function drive_ens(ens, qCell, yCell, hasQ)
%DRIVE_ENS Step an ensemble over a (q, y) stream honouring the hasQ pattern.
for t = 1:numel(yCell)
    if hasQ(t)
        ens.observe_q(qCell{t});
    end
    ens.observe_y(yCell{t});
end
end

function Ks = member_context_counts(members)
%MEMBER_CONTEXT_COUNTS Per-member instantiated-context count K (single-member
%   context_alignment; not the code under test).
R = numel(members.obj);
Ks = zeros(1, R);
for k = 1:R
    a = members.obj{k}.context_alignment();
    Ks(k) = a.K;
end
end

function v = naive_vector_mean(memberObjs, method)
%NAIVE_VECTOR_MEAN Plain equal-weight mean of a 1x(mc+1) probability vector.
%   Valid as the reference-frame aligned average only when every member shares
%   the same context count (so novel-slot positions coincide) -- used solely in
%   the trivial-alignment regime.
R = numel(memberObjs);
acc = memberObjs{1}.(method)();
for k = 2:R
    acc = acc + memberObjs{k}.(method)();
end
v = acc ./ R;
end

% ========================================================================= %
% ---------------------------- assert helpers ----------------------------- %
% ========================================================================= %

function assert_isequaln(name, actual, expected)
testutil.assertTrue(name, isequaln(actual, expected));
end

function k = map_sorted_keys(m)
if m.Count == 0
    k = [];
else
    k = sort(cell2mat(m.keys));
end
end

function assert_map_equal(name, mA, mB)
%ASSERT_MAP_EQUAL Same keys and bit-identical (isequaln) values.
testutil.assertTrue([name ' is a containers.Map (A)'], isa(mA, 'containers.Map'));
testutil.assertTrue([name ' is a containers.Map (B)'], isa(mB, 'containers.Map'));
kA = map_sorted_keys(mA);
kB = map_sorted_keys(mB);
testutil.assertTrue([name ' map keys equal'], isequal(kA, kB));
for i = 1:numel(kA)
    key = kA(i);
    testutil.assertTrue(sprintf('%s map value (key=%g)', name, key), ...
        isequaln(mA(key), mB(key)));
end
end

function assert_density_map_shape(name, m, Kref, K)
%ASSERT_DENSITY_MAP_SHAPE Keys subset of 1..Kref; each value a 1xK finite row.
testutil.assertTrue([name ' is a containers.Map'], isa(m, 'containers.Map'));
ks = map_sorted_keys(m);
for i = 1:numel(ks)
    key = ks(i);
    testutil.assertTrue(sprintf('%s key %g in 1..Kref', name, key), ...
        key >= 1 && key <= Kref && key == round(key));
    row = m(key);
    testutil.assertSize(sprintf('%s value shape (key=%g)', name, key), row, [1, K]);
    testutil.assertTrue(sprintf('%s value finite (key=%g)', name, key), ...
        all(isfinite(row)));
end
end

function assert_phase2_bit_equal(name, e1, e2, grid)
%ASSERT_PHASE2_BIT_EQUAL Bit-identical (isequaln) across every Phase-2 query.
assert_isequaln([name ' responsibilities_vector'], ...
    e1.responsibilities_vector(), e2.responsibilities_vector());
assert_isequaln([name ' predicted_context_probabilities_vector'], ...
    e1.predicted_context_probabilities_vector(), ...
    e2.predicted_context_probabilities_vector());
assert_isequaln([name ' sampled_context_count'], ...
    e1.sampled_context_count(), e2.sampled_context_count());
assert_isequaln([name ' stationary_context_probabilities'], ...
    e1.stationary_context_probabilities(), e2.stationary_context_probabilities());
assert_map_equal([name ' state_given_context_probability'], ...
    e1.state_given_context_probability(grid), ...
    e2.state_given_context_probability(grid));
assert_map_equal([name ' state_feedback_given_context_probability'], ...
    e1.state_feedback_given_context_probability(grid), ...
    e2.state_feedback_given_context_probability(grid));
end

function assert_trivial_density(name, ens, memberObjs, method, grid, tol)
%ASSERT_TRIVIAL_DENSITY Ensemble per-context density == naive key-by-key member
%   mean. In the trivial regime (K==1 for all members) reference label 1 is
%   present in every member, so the NaN-omit mean reduces to a plain mean over
%   all R members. Asserts key set == {1} and matching values.
mEns = ens.(method)(grid);
testutil.assertTrue([name ' is a containers.Map'], isa(mEns, 'containers.Map'));
testutil.assertTrue([name ' key set is {1}'], isequal(map_sorted_keys(mEns), 1));

R = numel(memberObjs);
acc = [];
cnt = 0;
for k = 1:R
    mk = memberObjs{k}.(method)(grid);
    if isKey(mk, 1)
        row = mk(1);
        if isempty(acc)
            acc = row;
        else
            acc = acc + row;
        end
        cnt = cnt + 1;
    end
end
testutil.assertTrue([name ' every member contributes context 1'], cnt == R);
naive = acc ./ cnt;
testutil.assertClose(mEns(1), naive, tol, [name ' == naive member mean']);
end

function close_all_safe()
% No figures are opened by these tests; kept for parity with other tests that
% may run in the same session.
end
