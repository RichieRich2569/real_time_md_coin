function test_ensemble_blindB
%TEST_ENSEMBLE Behavioural tests for RealTimeCOINEnsemble (blind, author B).
%
%   Authored strictly against docs/SPEC_ensemble.md (the contract), NOT against
%   any implementation. These tests build an independent oracle by driving R
%   seeded RealTimeCOIN members with the identical (q, y) stream and averaging
%   them by hand per SPEC section 5, reproducing the RNG substream contract of
%   SPEC section 3. Plain-function style to match the rest of tests/.
%
%   Spec ambiguities resolved here (documented for the reviewer):
%     * SPEC 3.1 permits "Threefry OR Philox" as the sub-streamable generator.
%       The exact-reproduction oracle must know which one. We DETECT it at
%       runtime (runs==1 ensemble vs a canonical substream-1 RealTimeCOIN built
%       with each candidate) and use the matching generator. If neither matches
%       the RNG contract is unmet and the affected tests fail by assertion.
%     * SPEC 5.4 "non-finite" is read as NaN OR Inf; the NaN-aware mean is
%       implemented as omitnan-mean after mapping Inf->NaN, so an all-non-finite
%       column yields NaN (matching COIN.weighted_sum_along_dimension).
%     * observe_q is always forwarded with the same q the ensemble receives;
%       a cue-free trial is q == NaN (or skipping observe_q entirely), and both
%       the ensemble and the oracle honour an identical hasQ pattern.

close_all_safe();

check_fanout_and_lockstep();
check_averaging_scalar();
check_averaging_multidim();
check_reproducibility();
check_executor_invariance_stepping();
check_executor_invariance_simulate();
check_seed_sensitivity_and_member_independence();
check_simulate_equivalence();
check_runs_one_reduction();
check_snapshot_roundtrip();
check_global_stream_untouched();

fprintf('test_ensemble: all ensemble contract checks passed.\n');
end

% ========================================================================= %
% Guarantee 1: fan-out and lockstep trial advance.
% ========================================================================= %
function check_fanout_and_lockstep()
mp = {'num_particles', 30, 'max_contexts', 4};
R = 4;
ens = RealTimeCOINEnsemble('runs', R, 'seed', 11, mp{:});
testutil.assertTrue('initial Trial is 0', ens.Trial == 0);

qCell = {1, 2, NaN, 1};
yCell = {0.2, -0.1, 0.05, NaN};
for t = 1:numel(yCell)
    ens.observe_q(qCell{t});
    ens.observe_y(yCell{t});
    testutil.assertTrue(sprintf('Trial advances by 1 (t=%d)', t), ens.Trial == t);
end

% runs/seed/weights accessors reflect construction; weights uniform (SPEC 2.2).
testutil.assertTrue('runs accessor', ens.runs == R);
testutil.assertTrue('seed accessor', ens.seed == 11);
testutil.assertClose(ens.weights, ones(1, R) / R, 0, 'weights must be uniform 1/R');

% Missing observation ([] and NaN) still advances a trial (SPEC 8).
ens.observe_y([]);
testutil.assertTrue('missing [] feedback advances trial', ens.Trial == numel(yCell) + 1);
end

% ========================================================================= %
% Guarantee 2: averaging correctness (scalar model). Core oracle.
% ========================================================================= %
function check_averaging_scalar()
% Small max_contexts is deliberate so the novel-context density saturates on
% some members (SPEC 8): a saturated member must contribute finite zeros, not
% NaN, to the averaged density.
mp = {'num_particles', 40, 'max_contexts', 3};
R = 5;
seed = 202;
N = 1;
gridState = linspace(-1, 1, 21);

qCell = {1, 1, 2, NaN, 2, 1, 2};
% hasQ(4) == false exercises the "never call observe_q" cue-free path.
hasQ  = [true, true, true, false, true, true, true];
yCell = {0.10, 0.25, NaN, 0.15, [], 0.30, 0.20};

gen = detect_generator(seed, mp, {1, 2}, {0.1, -0.05});
testutil.assertTrue('scalar: RNG generator (Threefry/Philox) detected', ~isempty(gen));

oracle = oracle_traces(gen, R, seed, mp, qCell, yCell, hasQ, N, gridState);

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
atol = 1e-9; rtol = 1e-9;
for t = 1:numel(yCell)
    if hasQ(t)
        ens.observe_q(qCell{t});
    end
    ens.observe_y(yCell{t});

    testutil.assertTrue(sprintf('scalar Trial (t=%d)', t), ens.Trial == t);

    u = ens.motor_output();
    testutil.assertSize(sprintf('scalar motor_output shape (t=%d)', t), u, [1 1]);
    assert_num_close(sprintf('scalar motor_output (t=%d)', t), u, oracle.motor{t}, atol, rtol);

    [mu, v] = ens.state_moments();
    assert_num_close(sprintf('scalar state mean (t=%d)', t), mu, oracle.mu{t}, atol, rtol);
    assert_num_close(sprintf('scalar state var  (t=%d)', t), v, oracle.v{t}, atol, rtol);
    testutil.assertTrue(sprintf('scalar var >= 0 (t=%d)', t), v >= 0);

    assert_num_close(sprintf('scalar state_probability (t=%d)', t), ...
        ens.state_probability(gridState), oracle.dstate{t}, atol, rtol);
    assert_num_close(sprintf('scalar state_feedback_probability (t=%d)', t), ...
        ens.state_feedback_probability(gridState), oracle.dsf{t}, atol, rtol);
    assert_num_close(sprintf('scalar novel_state_probability (t=%d)', t), ...
        ens.novel_state_probability(gridState), oracle.dnovel{t}, atol, rtol);
    assert_num_close(sprintf('scalar novel_state_feedback_probability (t=%d)', t), ...
        ens.novel_state_feedback_probability(gridState), oracle.dnovelf{t}, atol, rtol);

    % Densities must be entirely finite even where a member saturated its
    % context budget (zeros, not NaN) -- SPEC 8.
    testutil.assertTrue(sprintf('scalar novel density finite (t=%d)', t), ...
        all(isfinite(oracle.dnovel{t})));
end
end

% ========================================================================= %
% Guarantee 2 (cont): averaging correctness, multi-dimensional (state_dim==2).
% Exercises the pooled-mixture covariance formula (SPEC 5.2), which differs
% from a naive mean of per-run covariances whenever the per-run means differ.
% ========================================================================= %
function check_averaging_multidim()
mp = {'num_particles', 40, 'max_contexts', 4, 'state_dim', 2};
R = 4;
seed = 303;
N = 2;
gridState = [linspace(-0.4, 0.4, 11); linspace(-0.3, 0.3, 11)];

qCell = {1, 1, 2, 2, 1, 2};
hasQ  = [true, true, true, true, true, true];
yCell = {[0.10; 0.05], [0.20; -0.10], [NaN; 0.10], [0.15; 0.20], [], [0.05; 0.05]};

gen = detect_generator(seed, mp, {1, 2}, {[0.1; 0.0], [-0.05; 0.1]});
testutil.assertTrue('MD: RNG generator (Threefry/Philox) detected', ~isempty(gen));

oracle = oracle_traces(gen, R, seed, mp, qCell, yCell, hasQ, N, gridState);

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
atol = 1e-9; rtol = 1e-9;
for t = 1:numel(yCell)
    if hasQ(t)
        ens.observe_q(qCell{t});
    end
    ens.observe_y(yCell{t});

    u = ens.motor_output();
    testutil.assertSize(sprintf('MD motor_output shape (t=%d)', t), u, [N 1]);
    assert_num_close(sprintf('MD motor_output (t=%d)', t), u, oracle.motor{t}, atol, rtol);

    [mu, v] = ens.state_moments();
    testutil.assertSize(sprintf('MD mu shape (t=%d)', t), mu, [N 1]);
    testutil.assertSize(sprintf('MD v shape (t=%d)', t), v, [N N]);
    assert_num_close(sprintf('MD state mean (t=%d)', t), mu, oracle.mu{t}, atol, rtol);
    assert_num_close(sprintf('MD pooled covariance (t=%d)', t), v, oracle.v{t}, atol, rtol);
    % Covariance must be symmetric.
    assert_num_close(sprintf('MD covariance symmetric (t=%d)', t), v, v', 1e-12, 0);

    assert_num_close(sprintf('MD state_probability (t=%d)', t), ...
        ens.state_probability(gridState), oracle.dstate{t}, atol, rtol);
    assert_num_close(sprintf('MD state_feedback_probability (t=%d)', t), ...
        ens.state_feedback_probability(gridState), oracle.dsf{t}, atol, rtol);
    assert_num_close(sprintf('MD novel_state_probability (t=%d)', t), ...
        ens.novel_state_probability(gridState), oracle.dnovel{t}, atol, rtol);
    assert_num_close(sprintf('MD novel_state_feedback_probability (t=%d)', t), ...
        ens.novel_state_feedback_probability(gridState), oracle.dnovelf{t}, atol, rtol);
end
end

% ========================================================================= %
% Guarantee 3: reproducibility -- same (seed, runs) + same stream => identical.
% ========================================================================= %
function check_reproducibility()
mp = {'num_particles', 32, 'max_contexts', 4};
R = 4; seed = 77;
qCell = {1, 2, 1, NaN, 2, 1};
yCell = {0.1, -0.2, NaN, 0.05, [], 0.3};

a = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
b = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
for t = 1:numel(yCell)
    a.observe_q(qCell{t}); a.observe_y(yCell{t});
    b.observe_q(qCell{t}); b.observe_y(yCell{t});
    assert_all_queries_bit_equal(sprintf('reproducibility (t=%d)', t), a, b, ...
        1, linspace(-1, 1, 15));
end
end

% ========================================================================= %
% Guarantee 4: executor invariance (stepping) -- serial vs parallel and
% different segment_length must be bit-identical.
% ========================================================================= %
function check_executor_invariance_stepping()
mp = {'num_particles', 32, 'max_contexts', 4};
R = 4; seed = 91;
qCell = {1, 2, 2, 1, NaN, 2};
yCell = {0.15, -0.05, NaN, 0.2, 0.1, []};

serial   = RealTimeCOINEnsemble('runs', R, 'seed', seed, 'max_cores', 0, ...
    'segment_length', 1, mp{:});
parallel = RealTimeCOINEnsemble('runs', R, 'seed', seed, 'max_cores', 2, ...
    'segment_length', 3, mp{:});
for t = 1:numel(yCell)
    serial.observe_q(qCell{t});   serial.observe_y(yCell{t});
    parallel.observe_q(qCell{t}); parallel.observe_y(yCell{t});
    assert_all_queries_bit_equal(sprintf('executor invariance step (t=%d)', t), ...
        serial, parallel, 1, linspace(-1, 1, 15));
end
end

% ========================================================================= %
% Guarantee 4 (cont): executor invariance for simulate().
% ========================================================================= %
function check_executor_invariance_simulate()
mp = {'num_particles', 32, 'max_contexts', 4};
R = 4; seed = 91;
T = 6;
qSeq = [1 2 2 1 NaN 2];
ySeq = [0.15 -0.05 NaN 0.2 0.1 -0.1];

a = RealTimeCOINEnsemble('runs', R, 'seed', seed, 'max_cores', 0, ...
    'segment_length', 1, mp{:});
b = RealTimeCOINEnsemble('runs', R, 'seed', seed, 'max_cores', 2, ...
    'segment_length', 4, mp{:});
tra = a.simulate(qSeq, ySeq);
trb = b.simulate(qSeq, ySeq);

testutil.assertTrue('simulate motor_output executor-invariant', ...
    isequaln(tra.motor_output, trb.motor_output));
testutil.assertTrue('simulate state_mean executor-invariant', ...
    isequaln(tra.state_mean, trb.state_mean));
testutil.assertTrue('simulate state_var executor-invariant', ...
    isequaln(tra.state_var, trb.state_var));
testutil.assertTrue('simulate Trial vector', isequal(tra.Trial, 1:T));

% Output shape contract (SPEC 6): scalar model.
testutil.assertSize('simulate motor_output shape', tra.motor_output, [1 T]);
testutil.assertSize('simulate state_mean shape', tra.state_mean, [1 T]);
testutil.assertSize('simulate state_var shape (scalar)', tra.state_var, [1 T]);
end

% ========================================================================= %
% Guarantee 5: independence / seed sensitivity.
% ========================================================================= %
function check_seed_sensitivity_and_member_independence()
mp = {'num_particles', 40, 'max_contexts', 4};
R = 5;
qCell = {1, 2, 1, 2};
yCell = {0.2, -0.1, 0.15, 0.05};

e0 = RealTimeCOINEnsemble('runs', R, 'seed', 0, mp{:});
e1 = RealTimeCOINEnsemble('runs', R, 'seed', 123, mp{:});
for t = 1:numel(yCell)
    e0.observe_q(qCell{t}); e0.observe_y(yCell{t});
    e1.observe_q(qCell{t}); e1.observe_y(yCell{t});
end
testutil.assertTrue('different seeds give different motor output', ...
    abs(e0.motor_output() - e1.motor_output()) > 1e-6);

% Members within one ensemble follow different substreams => differ (a.s.).
seed = 55;
gen = detect_generator(seed, mp, {1, 2}, {0.1, -0.05});
testutil.assertTrue('seed-independence: generator detected', ~isempty(gen));
members = build_members(gen, R, seed, mp);
saved = RandStream.getGlobalStream();
restore = onCleanup(@() RandStream.setGlobalStream(saved));
for t = 1:numel(yCell)
    for k = 1:R
        RandStream.setGlobalStream(members.streams{k});
        members.obj{k}.observe_q(qCell{t});
        members.obj{k}.observe_y(yCell{t});
    end
end
RandStream.setGlobalStream(saved);
m1 = members.obj{1}.motor_output();
m2 = members.obj{2}.motor_output();
testutil.assertTrue('distinct members diverge', abs(m1 - m2) > 1e-9);
end

% ========================================================================= %
% Guarantee 6: simulate() equals the trial-by-trial stepping loop; repeatable;
% does not disturb live stepping state.
% ========================================================================= %
function check_simulate_equivalence()
mp = {'num_particles', 32, 'max_contexts', 4};
R = 4; seed = 404;
T = 7;
qSeq = [1 1 2 NaN 2 1 2];
ySeq = [0.10 0.25 NaN 0.15 0.0 0.30 0.20];

% Reference: trial-by-trial stepping, recording queries after each trial.
step = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
recMotor = zeros(1, T);
recMean = zeros(1, T);
recVar = zeros(1, T);
for t = 1:T
    step.observe_q(qSeq(t));
    step.observe_y(ySeq(:, t));
    recMotor(t) = step.motor_output();
    [mu, v] = step.state_moments();
    recMean(t) = mu;
    recVar(t) = v;
end

% simulate() on a fresh, identically-seeded ensemble must reproduce them.
sim = RealTimeCOINEnsemble('runs', R, 'seed', seed, mp{:});
tr = sim.simulate(qSeq, ySeq);
testutil.assertTrue('simulate motor == stepping', isequaln(tr.motor_output, recMotor));
testutil.assertTrue('simulate state_mean == stepping', isequaln(tr.state_mean, recMean));
testutil.assertTrue('simulate state_var == stepping', isequaln(tr.state_var, recVar));
testutil.assertTrue('simulate Trial vector', isequal(tr.Trial, 1:T));

% simulate() does not disturb the live stepping state of sim (SPEC 6).
testutil.assertTrue('simulate leaves live Trial untouched', sim.Trial == 0);

% simulate() is repeatable.
tr2 = sim.simulate(qSeq, ySeq);
testutil.assertTrue('simulate repeatable (motor)', isequaln(tr2.motor_output, tr.motor_output));
testutil.assertTrue('simulate repeatable (mean)', isequaln(tr2.state_mean, tr.state_mean));
testutil.assertTrue('simulate repeatable (var)', isequaln(tr2.state_var, tr.state_var));
end

% ========================================================================= %
% Guarantee 7: runs == 1 reduces exactly to the single seeded member.
% ========================================================================= %
function check_runs_one_reduction()
mp = {'num_particles', 40, 'max_contexts', 4};
seed = 606;
qCell = {1, 2, NaN, 1, 2};
yCell = {0.2, -0.1, 0.05, NaN, 0.3};
gridState = linspace(-1, 1, 17);

gen = detect_generator(seed, mp, {1, 2}, {0.1, -0.05});
testutil.assertTrue('runs==1: generator detected', ~isempty(gen));

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, mp{:});
members = build_members(gen, 1, seed, mp);
saved = RandStream.getGlobalStream();
restore = onCleanup(@() RandStream.setGlobalStream(saved));
for t = 1:numel(yCell)
    ens.observe_q(qCell{t});   % NaN cue forwarded verbatim as a cue-free trial
    ens.observe_y(yCell{t});

    RandStream.setGlobalStream(members.streams{1});
    members.obj{1}.observe_q(qCell{t});
    members.obj{1}.observe_y(yCell{t});
    RandStream.setGlobalStream(saved);

    single = members.obj{1};
    testutil.assertTrue(sprintf('runs==1 motor (t=%d)', t), ...
        isequaln(ens.motor_output(), single.motor_output()));
    [emu, ev] = ens.state_moments();
    [smu, sv] = single.state_moments();
    testutil.assertTrue(sprintf('runs==1 mu (t=%d)', t), isequaln(emu, smu));
    testutil.assertTrue(sprintf('runs==1 var (t=%d)', t), isequaln(ev, sv));
    testutil.assertTrue(sprintf('runs==1 state_probability (t=%d)', t), ...
        isequaln(ens.state_probability(gridState), single.state_probability(gridState)));
    testutil.assertTrue(sprintf('runs==1 novel_state_feedback (t=%d)', t), ...
        isequaln(ens.novel_state_feedback_probability(gridState), ...
                 single.novel_state_feedback_probability(gridState)));
end
end

% ========================================================================= %
% Guarantee 8: snapshot / loadSnapshot round-trip on RealTimeCOIN (SPEC 7).
% ========================================================================= %
function check_snapshot_roundtrip()
mp = {'num_particles', 40, 'max_contexts', 4, 'infer_bias', true};

rng(7);
b = RealTimeCOIN(mp{:});
drives_q = {1, 2, 1, 2};
drives_y = {0.2, -0.1, 0.15, 0.05};
for t = 1:numel(drives_y)
    b.observe_q(drives_q{t});
    b.observe_y(drives_y{t});
end

s = b.snapshot();                 % additive public method (SPEC 7)
a = RealTimeCOIN(mp{:});
a.loadSnapshot(s);                % restore into a fresh object

grid = linspace(-1, 1, 21);
% Immediately after restore, a must reproduce b's query outputs exactly.
testutil.assertTrue('snapshot Trial', a.Trial == b.Trial);
testutil.assertTrue('snapshot motor', isequaln(a.motor_output(), b.motor_output()));
[amu, av] = a.state_moments();
[bmu, bv] = b.state_moments();
testutil.assertTrue('snapshot mu', isequaln(amu, bmu));
testutil.assertTrue('snapshot var', isequaln(av, bv));
testutil.assertTrue('snapshot state density', ...
    isequaln(a.state_probability(grid), b.state_probability(grid)));
testutil.assertTrue('snapshot feedback density', ...
    isequaln(a.state_feedback_probability(grid), b.state_feedback_probability(grid)));

% A matched-RNG future step must keep them identical (state fully restored).
saved = RandStream.getGlobalStream();
restore = onCleanup(@() RandStream.setGlobalStream(saved));
RandStream.setGlobalStream(RandStream('Threefry', 'Seed', 321));
b.observe_q(1); b.observe_y(0.12);
RandStream.setGlobalStream(RandStream('Threefry', 'Seed', 321));
a.observe_q(1); a.observe_y(0.12);
RandStream.setGlobalStream(saved);
testutil.assertTrue('snapshot post-step motor', isequaln(a.motor_output(), b.motor_output()));
testutil.assertTrue('snapshot post-step density', ...
    isequaln(a.state_probability(grid), b.state_probability(grid)));
end

% ========================================================================= %
% Guarantee 9: the ensemble leaves the caller's global RNG stream unchanged
% after construction, stepping, and simulate (SPEC 3.5).
% ========================================================================= %
function check_global_stream_untouched()
mp = {'num_particles', 30, 'max_contexts', 4};

% Fix a known global stream so we can compare exact internal state.
saved = RandStream.getGlobalStream();
restore = onCleanup(@() RandStream.setGlobalStream(saved));
gs = RandStream('Threefry', 'Seed', 12345);
RandStream.setGlobalStream(gs);
stateBefore = gs.State;

ens = RealTimeCOINEnsemble('runs', 3, 'seed', 5, mp{:});
testutil.assertTrue('construction keeps same global stream handle', ...
    RandStream.getGlobalStream() == gs);
testutil.assertTrue('construction leaves global stream state', ...
    isequal(gs.State, stateBefore));

ens.observe_q(1); ens.observe_y(0.1);
testutil.assertTrue('observe keeps same global stream handle', ...
    RandStream.getGlobalStream() == gs);
testutil.assertTrue('observe leaves global stream state', ...
    isequal(gs.State, stateBefore));

ens.simulate([1 2 1], [0.1 -0.1 0.05]);
testutil.assertTrue('simulate keeps same global stream handle', ...
    RandStream.getGlobalStream() == gs);
testutil.assertTrue('simulate leaves global stream state', ...
    isequal(gs.State, stateBefore));

RandStream.setGlobalStream(saved);
end

% ========================================================================= %
% ---------------------------- oracle helpers ----------------------------- %
% ========================================================================= %

function gen = detect_generator(seed, memberParams, qFirst, yFirst)
%DETECT_GENERATOR Identify which sub-streamable generator the ensemble uses.
%   Drives a runs==1 ensemble over a short (q,y) prefix and compares its
%   motor_output against a canonical substream-1 RealTimeCOIN built with each
%   candidate generator. Returns the matching generator name, or '' if none
%   matches (RNG contract SPEC 3 unmet -- e.g. against the NaN stub).
cands = {'Threefry', 'Philox'};
gen = '';

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, memberParams{:});
for t = 1:numel(yFirst)
    ens.observe_q(qFirst{t});
    ens.observe_y(yFirst{t});
end
target = ens.motor_output();
if any(~isfinite(target(:)))
    return;   % stub (NaN) or degenerate -- cannot detect
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
%   family (SPEC 3.1). Returns a struct with .obj (members) and .streams (the
%   persistent RandStream objects, to be reactivated around each member step).
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

function tr = oracle_traces(gen, runs, seed, memberParams, qCell, yCell, hasQ, N, gridState)
%ORACLE_TRACES Independent per-trial averaged queries built from R members.
%   Reproduces the ensemble RNG contract (SPEC 3) exactly and averages per
%   SPEC 5 (mean motor, pooled-mixture moments, mean density). Query methods
%   draw no randomness, so they are read out with the global stream restored.
saved = RandStream.getGlobalStream();
restore = onCleanup(@() RandStream.setGlobalStream(saved));

members = build_members(gen, runs, seed, memberParams);

T = numel(yCell);
tr = struct();
[tr.motor, tr.mu, tr.v, tr.dstate, tr.dsf, tr.dnovel, tr.dnovelf] = deal(cell(1, T));
for t = 1:T
    for k = 1:runs
        RandStream.setGlobalStream(members.streams{k});
        if hasQ(t)
            members.obj{k}.observe_q(qCell{t});
        end
        members.obj{k}.observe_y(yCell{t});
    end
    RandStream.setGlobalStream(saved);

    tr.motor{t} = avg_motor(members.obj, N);
    [tr.mu{t}, tr.v{t}] = avg_moments(members.obj, N);
    tr.dstate{t}  = avg_density(members.obj, 'state_probability', gridState);
    tr.dsf{t}     = avg_density(members.obj, 'state_feedback_probability', gridState);
    tr.dnovel{t}  = avg_density(members.obj, 'novel_state_probability', gridState);
    tr.dnovelf{t} = avg_density(members.obj, 'novel_state_feedback_probability', gridState);
end
RandStream.setGlobalStream(saved);
end

function u = avg_motor(members, N)
R = numel(members);
M = zeros(N, R);
for k = 1:R
    M(:, k) = members{k}.motor_output();
end
u = nan_aware_mean(M, 2);
end

function [mu, v] = avg_moments(members, N)
%AVG_MOMENTS Equal-weight pooled-mixture moments (SPEC 5.2).
R = numel(members);
MU = zeros(N, R);
if N == 1
    Tk = zeros(1, R);
else
    Tk = zeros(N, N, R);
end
for k = 1:R
    [muk, vk] = members{k}.state_moments();
    MU(:, k) = muk;
    if N == 1
        Tk(k) = vk + muk^2;
    else
        Tk(:, :, k) = vk + (muk * muk');
    end
end
mu = nan_aware_mean(MU, 2);
if N == 1
    meanT = nan_aware_mean(Tk, 2);
    v = max(meanT - mu^2, 0);
else
    meanT = nan_aware_mean(Tk, 3);
    v = meanT - (mu * mu');
    v = (v + v') ./ 2;
end
end

function d = avg_density(members, method, values)
R = numel(members);
first = members{1}.(method)(values);
K = numel(first);
D = zeros(R, K);
D(1, :) = first(:)';
for k = 2:R
    dk = members{k}.(method)(values);
    D(k, :) = dk(:)';
end
d = nan_aware_mean(D, 1);
end

function m = nan_aware_mean(x, dim)
%NAN_AWARE_MEAN omitnan-mean with all-non-finite -> NaN (SPEC 5.4).
x(~isfinite(x)) = NaN;
m = mean(x, dim, 'omitnan');
end

% ========================================================================= %
% ---------------------------- assert helpers ----------------------------- %
% ========================================================================= %

function assert_num_close(name, actual, expected, atol, rtol)
%ASSERT_NUM_CLOSE Elementwise closeness with matching NaN pattern.
if ~isequal(size(actual), size(expected))
    error('test_ensemble:size', 'FAILED: %s size [%s] != [%s]', ...
        name, num2str(size(actual)), num2str(size(expected)));
end
a = actual(:); b = expected(:);
if any(isnan(a) ~= isnan(b))
    error('test_ensemble:nanPattern', 'FAILED: %s NaN pattern mismatch', name);
end
fin = ~isnan(a) & ~isnan(b);
d = abs(a(fin) - b(fin));
tol = atol + rtol .* max(abs(a(fin)), abs(b(fin)));
if ~all(d <= tol)
    error('test_ensemble:tol', 'FAILED: %s max abs err %.3g exceeds tolerance', ...
        name, max([0; d]));
end
end

function assert_all_queries_bit_equal(name, e1, e2, N, gridState)
%ASSERT_ALL_QUERIES_BIT_EQUAL Bit-identical outputs across every query method.
testutil.assertTrue([name ' Trial'], e1.Trial == e2.Trial);
testutil.assertTrue([name ' motor_output'], isequaln(e1.motor_output(), e2.motor_output()));
[m1, v1] = e1.state_moments();
[m2, v2] = e2.state_moments();
testutil.assertTrue([name ' state mean'], isequaln(m1, m2));
testutil.assertTrue([name ' state var'], isequaln(v1, v2));
if N == 1
    g = gridState;
else
    g = repmat(gridState, N, 1);
end
testutil.assertTrue([name ' state_probability'], ...
    isequaln(e1.state_probability(g), e2.state_probability(g)));
testutil.assertTrue([name ' state_feedback_probability'], ...
    isequaln(e1.state_feedback_probability(g), e2.state_feedback_probability(g)));
testutil.assertTrue([name ' novel_state_probability'], ...
    isequaln(e1.novel_state_probability(g), e2.novel_state_probability(g)));
testutil.assertTrue([name ' novel_state_feedback_probability'], ...
    isequaln(e1.novel_state_feedback_probability(g), e2.novel_state_feedback_probability(g)));
end

function close_all_safe()
% No figures are opened by these tests; kept for parity with other tests that
% may run in the same session.
end
