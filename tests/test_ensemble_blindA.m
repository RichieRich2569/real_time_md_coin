function test_ensemble_blindA
%TEST_ENSEMBLE Blind behavioural tests for RealTimeCOINEnsemble.
%
%   Authored strictly against docs/SPEC_ensemble.md (the contract), NOT against
%   any implementation. The independent oracle is built here by constructing R
%   ordinary RealTimeCOIN members, driving them with the identical (q, y)
%   stream, and averaging per SPEC section 5 -- while replicating the RNG
%   substream contract of SPEC section 3 so the reference lines up bit-for-bit
%   with the ensemble.
%
%   These tests intentionally FAIL against the signature-only stub (which
%   returns NaN); they are the oracle for the real implementation.
%
%   Coverage maps to SPEC section 9 "testable guarantees":
%     (1) fan-out and lockstep trial counting
%     (2) averaging correctness (motor_output, pooled state_moments, densities)
%     (3) reproducibility (same seed/runs/stream => bit-identical)
%     (4) executor invariance (max_cores / segment_length independence)
%     (5) independence / seed sensitivity
%     (6) simulate == stepping loop; simulate repeatable and non-disturbing
%     (7) runs == 1 reduction to the single member
%     (8) snapshot/loadSnapshot round-trip on RealTimeCOIN
%     (9) global RNG left unchanged; edge cases (missing / cue-free trials)
%
%   Documented spec ambiguities / interpretations (see SPEC section 3.1):
%     * Generator family: SPEC says "e.g. Threefry or Philox". The substream
%       generator is not uniquely pinned, so the oracle auto-detects which of
%       {Threefry, Philox} reproduces the ensemble (both are spec-permitted) and
%       then holds every other query to exact agreement under that generator. If
%       NEITHER reproduces the ensemble, that is a contract violation and the
%       test errors -- this does not weaken the correctness assertion.
%     * Stream indexing: member k (k = 1..R) uses StreamIndices = k, i.e.
%       1-based, matching MATLAB's requirement that StreamIndices lie in
%       1..NumStreams (0-based is impossible), and NumStreams = runs.

TOL = 1e-9;   % RNG-matched averaging: allow only summation-order slack.

check_fanout_and_lockstep(TOL);
check_averaging_scalar(TOL);
check_averaging_multidim(TOL);
check_reproducibility();
check_executor_invariance();
check_seed_sensitivity();
check_simulate_matches_stepping(TOL);
check_simulate_repeatable_and_nondisturbing(TOL);
check_runs_one_reduction(TOL);
check_snapshot_roundtrip();
check_global_rng_unchanged();
check_edge_cases(TOL);
end

% ------------------------------------------------------------------------
% (1) Fan-out and lockstep
% ------------------------------------------------------------------------
function check_fanout_and_lockstep(TOL)
p = scalarParams();
R = 4; seed = 11;
[qSeq, ySeq] = scalarStream();
gen = detectGenerator(seed, R, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, R, p);
orig = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(orig));

testutil.assertTrue('Trial starts at 0', ens.Trial == 0);
for t = 1:numel(ySeq)
    ens.observe_q(qSeq(t));
    ens.observe_y(ySeq(t));
    oracleStep(members, streams, qSeq(t), ySeq(t));
    % lockstep: exactly one trial advanced per observe_y
    testutil.assertTrue(sprintf('Trial==%d', t), ens.Trial == t);
    % fan-out: identical (q,y) reached every member => average matches oracle
    assertNear(sprintf('fanout motor t=%d', t), ...
        ens.motor_output(), oracleMotor(members), TOL);
end
end

% ------------------------------------------------------------------------
% (2) Averaging correctness -- scalar model
% ------------------------------------------------------------------------
function check_averaging_scalar(TOL)
p = scalarParams();
R = 4; seed = 23;
[qSeq, ySeq] = scalarStream();
grid = linspace(-1.2, 1.2, 9);
gen = detectGenerator(seed, R, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, R, p);
orig = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(orig));

for t = 1:numel(ySeq)
    ens.observe_q(qSeq(t));
    ens.observe_y(ySeq(t));
    oracleStep(members, streams, qSeq(t), ySeq(t));

    % motor_output = mean over runs
    assertNear('scalar motor_output', ens.motor_output(), oracleMotor(members), TOL);

    % state_moments = pooled-mixture (law of total variance), NOT mean of vars
    [mu, v] = ens.state_moments();
    [muRef, vRef] = oracleMoments(members, 1);
    assertNear('scalar state_moments mu', mu, muRef, TOL);
    assertNear('scalar state_moments v', v, vRef, TOL);
    testutil.assertTrue('scalar variance nonnegative', v >= 0);

    % density queries = mean of per-run densities
    for name = {'state_probability', 'state_feedback_probability', ...
            'novel_state_probability', 'novel_state_feedback_probability'}
        m = name{1};
        d = ens.(m)(grid);
        dRef = oracleDensity(members, m, grid);
        testutil.assertSize(['scalar ' m ' shape'], d, [1, numel(grid)]);
        assertNear(['scalar ' m], d, dRef, TOL);
    end
end
end

% ------------------------------------------------------------------------
% (2) Averaging correctness -- multi-dimensional model (state_dim == 2)
% ------------------------------------------------------------------------
function check_averaging_multidim(TOL)
p = mdParams();
R = 4; seed = 37;
[qSeq, ySeq] = mdStream();
grid = [linspace(-1, 1, 6); linspace(-0.5, 0.5, 6)];   % 2xK columns
gen = detectGenerator(seed, R, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, R, p);
orig = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(orig));

for t = 1:size(ySeq, 2)
    ens.observe_q(qSeq(t));
    ens.observe_y(ySeq(:, t));
    oracleStep(members, streams, qSeq(t), ySeq(:, t));

    u = ens.motor_output();
    testutil.assertSize('MD motor_output shape', u, [2, 1]);
    assertNear('MD motor_output', u, oracleMotor(members), TOL);

    [mu, v] = ens.state_moments();
    [muRef, vRef] = oracleMoments(members, 2);
    testutil.assertSize('MD mu shape', mu, [2, 1]);
    testutil.assertSize('MD v shape', v, [2, 2]);
    assertNear('MD state_moments mu', mu, muRef, TOL);
    assertNear('MD state_moments v', v, vRef, TOL);
    % pooled covariance must be symmetric (SPEC 5.2)
    assertNear('MD covariance symmetric', v, v', TOL);

    for name = {'state_probability', 'state_feedback_probability', ...
            'novel_state_probability', 'novel_state_feedback_probability'}
        m = name{1};
        d = ens.(m)(grid);
        dRef = oracleDensity(members, m, grid);
        testutil.assertSize(['MD ' m ' shape'], d, [1, size(grid, 2)]);
        assertNear(['MD ' m], d, dRef, TOL);
    end
end
end

% ------------------------------------------------------------------------
% (3) Reproducibility: same (seed, runs) + same stream => bit-identical
% ------------------------------------------------------------------------
function check_reproducibility()
p = scalarParams();
R = 5; seed = 101;
[qSeq, ySeq] = scalarStream();
grid = linspace(-1, 1, 7);

ensA = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
ensB = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
for t = 1:numel(ySeq)
    ensA.observe_q(qSeq(t)); ensA.observe_y(ySeq(t));
    ensB.observe_q(qSeq(t)); ensB.observe_y(ySeq(t));
    assertBit('reproducible motor_output', ensA.motor_output(), ensB.motor_output());
    [muA, vA] = ensA.state_moments();
    [muB, vB] = ensB.state_moments();
    assertBit('reproducible mu', muA, muB);
    assertBit('reproducible v', vA, vB);
    assertBit('reproducible density', ensA.state_probability(grid), ensB.state_probability(grid));
end
end

% ------------------------------------------------------------------------
% (4) Executor invariance: serial vs parallel, any segment_length, identical
% ------------------------------------------------------------------------
function check_executor_invariance()
p = scalarParams();
R = 5; seed = 202;
[qSeq, ySeq] = scalarStream();
grid = linspace(-1, 1, 7);

% Stepping path.
ensSerial = RealTimeCOINEnsemble('runs', R, 'seed', seed, ...
    'max_cores', 0, 'segment_length', 1, p{:});
ensPar = RealTimeCOINEnsemble('runs', R, 'seed', seed, ...
    'max_cores', 2, 'segment_length', 4, p{:});
for t = 1:numel(ySeq)
    ensSerial.observe_q(qSeq(t)); ensSerial.observe_y(ySeq(t));
    ensPar.observe_q(qSeq(t));    ensPar.observe_y(ySeq(t));
    assertBit('executor motor_output', ensSerial.motor_output(), ensPar.motor_output());
    [mS, vS] = ensSerial.state_moments();
    [mP, vP] = ensPar.state_moments();
    assertBit('executor mu', mS, mP);
    assertBit('executor v', vS, vP);
    assertBit('executor density', ensSerial.state_probability(grid), ensPar.state_probability(grid));
end

% simulate path (SPEC 6 obeys executor invariance too).
ensS2 = RealTimeCOINEnsemble('runs', R, 'seed', seed, ...
    'max_cores', 0, 'segment_length', 1, p{:});
ensP2 = RealTimeCOINEnsemble('runs', R, 'seed', seed, ...
    'max_cores', 3, 'segment_length', 5, p{:});
tracesS = ensS2.simulate(qSeq, ySeq);
tracesP = ensP2.simulate(qSeq, ySeq);
assertBit('executor simulate motor', tracesS.motor_output, tracesP.motor_output);
assertBit('executor simulate state_mean', tracesS.state_mean, tracesP.state_mean);
assertBit('executor simulate state_var', tracesS.state_var, tracesP.state_var);
end

% ------------------------------------------------------------------------
% (5) Independence / seed sensitivity
% ------------------------------------------------------------------------
function check_seed_sensitivity()
p = scalarParams();
R = 4;
[qSeq, ySeq] = scalarStream();

% Different seeds diverge.
ens1 = RealTimeCOINEnsemble('runs', R, 'seed', 1, p{:});
ens2 = RealTimeCOINEnsemble('runs', R, 'seed', 999, p{:});
for t = 1:numel(ySeq)
    ens1.observe_q(qSeq(t)); ens1.observe_y(ySeq(t));
    ens2.observe_q(qSeq(t)); ens2.observe_y(ySeq(t));
end
testutil.assertTrue('different seeds diverge', ...
    abs(ens1.motor_output() - ens2.motor_output()) > 1e-6);

% Members within one ensemble are mutually independent: distinct substreams
% (k=1 vs k=2) yield distinct member trajectories under identical observations.
gen = detectGenerator(1, R, p, qSeq, ySeq);
[members, streams] = oracleBuild(gen, 1, R, p);
orig = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(orig));
for t = 1:numel(ySeq)
    oracleStep(members, streams, qSeq(t), ySeq(t));
end
testutil.assertTrue('members within ensemble differ', ...
    abs(members{1}.motor_output() - members{2}.motor_output()) > 1e-9);
end

% ------------------------------------------------------------------------
% (6a) simulate == trial-by-trial stepping loop reading same queries
% ------------------------------------------------------------------------
function check_simulate_matches_stepping(TOL)
p = scalarParams();
R = 4; seed = 303;
[qSeq, ySeq] = scalarStream();
T = numel(ySeq);

ensBatch = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
traces = ensBatch.simulate(qSeq, ySeq);

testutil.assertSize('simulate motor_output shape', traces.motor_output, [1, T]);
testutil.assertSize('simulate state_mean shape', traces.state_mean, [1, T]);
testutil.assertSize('simulate state_var shape', traces.state_var, [1, T]);
testutil.assertTrue('simulate Trial vector', isequal(traces.Trial, 1:T));

ensStep = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
for t = 1:T
    ensStep.observe_q(qSeq(t));
    ensStep.observe_y(ySeq(t));
    [mu, v] = ensStep.state_moments();
    assertNear(sprintf('simulate==step motor t=%d', t), ...
        traces.motor_output(:, t), ensStep.motor_output(), TOL);
    assertNear(sprintf('simulate==step mean t=%d', t), ...
        traces.state_mean(:, t), mu, TOL);
    assertNear(sprintf('simulate==step var t=%d', t), ...
        traces.state_var(:, t), v, TOL);
end
end

% ------------------------------------------------------------------------
% (6b) simulate repeatable; does not disturb live stepping state
% ------------------------------------------------------------------------
function check_simulate_repeatable_and_nondisturbing(TOL)
p = scalarParams();
R = 4; seed = 404;
[qSeq, ySeq] = scalarStream();

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});

% Two simulate calls on the same ensemble yield identical traces.
t1 = ens.simulate(qSeq, ySeq);
t2 = ens.simulate(qSeq, ySeq);
assertBit('simulate repeatable motor', t1.motor_output, t2.motor_output);
assertBit('simulate repeatable mean', t1.state_mean, t2.state_mean);
assertBit('simulate repeatable var', t1.state_var, t2.state_var);

% simulate must not disturb the live stepping state of ens.
ensLive = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
for t = 1:3
    ensLive.observe_q(qSeq(t)); ensLive.observe_y(ySeq(t));
end
moBefore = ensLive.motor_output();
[muBefore, vBefore] = ensLive.state_moments();
trialBefore = ensLive.Trial;
ensLive.simulate(qSeq, ySeq);    % one-shot batch on a fresh member set
testutil.assertTrue('simulate leaves Trial', ensLive.Trial == trialBefore);
[muAfter, vAfter] = ensLive.state_moments();
assertNear('simulate leaves motor', ensLive.motor_output(), moBefore, TOL);
assertNear('simulate leaves mu', muAfter, muBefore, TOL);
assertNear('simulate leaves v', vAfter, vBefore, TOL);
end

% ------------------------------------------------------------------------
% (7) runs == 1 reduces to the single member exactly
% ------------------------------------------------------------------------
function check_runs_one_reduction(TOL)
p = scalarParams();
seed = 55;
[qSeq, ySeq] = scalarStream();
grid = linspace(-1, 1, 7);
gen = detectGenerator(seed, 1, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, 1, p);
orig = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(orig));

for t = 1:numel(ySeq)
    ens.observe_q(qSeq(t)); ens.observe_y(ySeq(t));
    oracleStep(members, streams, qSeq(t), ySeq(t));
    member = members{1};
    assertNear('runs1 motor', ens.motor_output(), member.motor_output(), TOL);
    [mu, v] = ens.state_moments();
    [muM, vM] = member.state_moments();
    assertNear('runs1 mu', mu, muM, TOL);
    assertNear('runs1 v', v, vM, TOL);
    assertNear('runs1 density', ens.state_probability(grid), member.state_probability(grid), TOL);
end
end

% ------------------------------------------------------------------------
% (8) snapshot/loadSnapshot round-trip on RealTimeCOIN (SPEC section 7)
% ------------------------------------------------------------------------
function check_snapshot_roundtrip()
p = scalarParams();
grid = linspace(-1, 1, 7);

a = RealTimeCOIN(p{:});
qSeq = [1 1 2 2 1];
ySeq = [0.1 0.2 -0.1 0.05 0.15];
for t = 1:numel(ySeq)
    a.observe_q(qSeq(t)); a.observe_y(ySeq(t));
end

s = a.snapshot();
b = RealTimeCOIN(p{:});
b.loadSnapshot(s);

% Immediately after load: identical query outputs (no new randomness).
testutil.assertTrue('snapshot Trial', b.Trial == a.Trial);
assertNear('snapshot motor', b.motor_output(), a.motor_output(), 0);
[muA, vA] = a.state_moments();
[muB, vB] = b.state_moments();
assertNear('snapshot mu', muB, muA, 0);
assertNear('snapshot v', vB, vA, 0);
assertNear('snapshot density', b.state_probability(grid), a.state_probability(grid), 0);

% For the SAME subsequent input under the SAME RNG state, b reproduces a.
rs = RandStream('Threefry', 'Seed', 7777);
orig = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(orig));
RandStream.setGlobalStream(rs);
savedState = rs.State;
a.observe_q(2); a.observe_y(0.3);
rs.State = savedState;      % rewind so b sees identical randomness
b.observe_q(2); b.observe_y(0.3);
assertNear('snapshot post-step motor', b.motor_output(), a.motor_output(), 0);
[muA2, vA2] = a.state_moments();
[muB2, vB2] = b.state_moments();
assertNear('snapshot post-step mu', muB2, muA2, 0);
assertNear('snapshot post-step v', vB2, vA2, 0);
end

% ------------------------------------------------------------------------
% (9a) Ensemble leaves the caller's global RNG stream unchanged (SPEC 3.5)
% ------------------------------------------------------------------------
function check_global_rng_unchanged()
p = scalarParams();
[qSeq, ySeq] = scalarStream();

rng(123456);
before = rng;

ens = RealTimeCOINEnsemble('runs', 4, 'seed', 9, p{:});
assertRngUnchanged('after construction', before);

ens.observe_q(qSeq(1));
ens.observe_y(ySeq(1));
assertRngUnchanged('after observe_y', before);

ens.simulate(qSeq, ySeq);
assertRngUnchanged('after simulate', before);
end

% ------------------------------------------------------------------------
% (9b) Edge cases: missing observation, cue-free trials (SPEC section 8)
% ------------------------------------------------------------------------
function check_edge_cases(TOL)
% Scalar: mix in a missing observation (NaN feedback) and a cue-free trial.
p = scalarParams();
R = 4; seed = 71;
qSeq = {1, [], 2, 1};         % [] cue-free trial
ySeq = {0.1, 0.2, NaN, 0.15}; % NaN missing observation
grid = linspace(-1, 1, 7);
gen = detectGeneratorCell(seed, R, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, R, p);
orig = RandStream.getGlobalStream;
cleanup = onCleanup(@() RandStream.setGlobalStream(orig));

for t = 1:numel(ySeq)
    q = qSeq{t}; y = ySeq{t};
    if isempty(q)
        % cue-free: forward the cue-free path to all members
        ens.observe_q([]);
    else
        ens.observe_q(q);
    end
    ens.observe_y(y);
    oracleStep(members, streams, q, y);

    testutil.assertTrue(sprintf('edge Trial==%d', t), ens.Trial == t);
    u = ens.motor_output();
    assertNear(sprintf('edge motor t=%d', t), u, oracleMotor(members), TOL);
    testutil.assertTrue(sprintf('edge motor finite t=%d', t), all(isfinite(u)));

    d = ens.novel_state_probability(grid);
    % Novel-context density stays a finite, well-defined contribution even at
    % saturation (zeros, not NaN) -- SPEC section 8.
    testutil.assertTrue(sprintf('edge novel density finite t=%d', t), all(isfinite(d)));
    assertNear(sprintf('edge novel density t=%d', t), d, ...
        oracleDensity(members, 'novel_state_probability', grid), TOL);
end
end

% ========================================================================
% Oracle helpers -- replicate SPEC section 3 RNG contract + section 5 averaging
% ========================================================================

function p = scalarParams()
p = {'num_particles', 40, 'max_contexts', 3, 'infer_bias', true};
end

function p = mdParams()
p = {'state_dim', 2, 'num_particles', 40, 'max_contexts', 3};
end

function [qSeq, ySeq] = scalarStream()
qSeq = [1 1 2 2 1 2];
ySeq = [0.10 0.25 -0.10 0.30 0.05 -0.20];
end

function [qSeq, ySeq] = mdStream()
qSeq = [1 1 2 2 1 2];
ySeq = [0.10  0.25 -0.10  0.30  0.05 -0.20; ...
        -0.05 0.15  0.20 -0.10  0.00  0.25];
end

function [members, streams] = oracleBuild(gen, seed, R, memberParams)
%ORACLEBUILD Construct R RealTimeCOIN members, each under its own substream.
%   Member k is constructed while member k's substream (StreamIndices = k of
%   NumStreams = R, Seed = seed) is the global stream, so that construction
%   randomness derives from substream k (SPEC 3.1). The caller's global stream
%   is restored on return.
orig = RandStream.getGlobalStream;
restore = onCleanup(@() RandStream.setGlobalStream(orig));
members = cell(1, R);
streams = cell(1, R);
for k = 1:R
    streams{k} = RandStream.create(gen, 'NumStreams', R, 'StreamIndices', k, 'Seed', seed);
    RandStream.setGlobalStream(streams{k});
    members{k} = RealTimeCOIN(memberParams{:});
end
end

function oracleStep(members, streams, q, y)
%ORACLESTEP Drive every member with the identical (q, y) under its substream.
orig = RandStream.getGlobalStream;
restore = onCleanup(@() RandStream.setGlobalStream(orig));
for k = 1:numel(members)
    RandStream.setGlobalStream(streams{k});
    members{k}.observe_q(q);   % draws no randomness (SPEC 4.1)
    members{k}.observe_y(y);
end
end

function u = oracleMotor(members)
%ORACLEMOTOR Equal-weight, NaN-aware mean of per-member motor_output.
R = numel(members);
first = members{1}.motor_output();
N = numel(first);
X = zeros(N, R);
X(:, 1) = first;
for k = 2:R
    X(:, k) = members{k}.motor_output();
end
u = nanmeanFinite(X);
end

function [mu, v] = oracleMoments(members, N)
%ORACLEMOMENTS Pooled equal-weight mixture moments (SPEC 5.2), NaN-aware.
R = numel(members);
MU = zeros(N, R);
if N == 1
    SEC = zeros(1, R);
    for k = 1:R
        [mk, vk] = members{k}.state_moments();
        MU(:, k) = mk;
        SEC(k) = vk + mk.^2;
    end
    mu = nanmeanFinite(MU);
    second = nanmeanFinite(SEC);       % scalar mean over runs
    v = max(second - mu.^2, 0);
else
    SEC = zeros(N, N, R);
    for k = 1:R
        [mk, vk] = members{k}.state_moments();
        MU(:, k) = mk;
        SEC(:, :, k) = vk + (mk * mk');
    end
    mu = nanmeanFinite(MU);
    second = zeros(N, N);
    for i = 1:N
        for j = 1:N
            second(i, j) = nanmeanFinite(reshape(SEC(i, j, :), 1, R));
        end
    end
    v = second - (mu * mu');
    v = (v + v') ./ 2;
end
end

function d = oracleDensity(members, method, values)
%ORACLEDENSITY Equal-weight, NaN-aware mean of per-member densities (SPEC 5.3).
R = numel(members);
first = members{1}.(method)(values);
K = numel(first);
D = zeros(R, K);
D(1, :) = first;
for k = 2:R
    D(k, :) = members{k}.(method)(values);
end
d = zeros(1, K);
for j = 1:K
    d(j) = nanmeanFinite(D(:, j)');
end
end

function m = nanmeanFinite(X)
%NANMEANFINITE Row-wise mean over finite entries; NaN if a row is all non-finite.
%   Implements the SPEC 5.4 rule: average over the runs where the value is
%   finite; if every run is non-finite the entry is NaN.
if isrow(X) || iscolumn(X)
    row = X(:)';
    fin = isfinite(row);
    if any(fin)
        m = mean(row(fin));
    else
        m = NaN;
    end
    return;
end
m = nan(size(X, 1), 1);
for i = 1:size(X, 1)
    row = X(i, :);
    fin = isfinite(row);
    if any(fin)
        m(i) = mean(row(fin));
    end
end
end

function gen = detectGenerator(seed, R, memberParams, qSeq, ySeq)
%DETECTGENERATOR Identify which spec-permitted substream generator the ensemble
%   uses, by checking motor_output agreement over the stream. Errors if neither
%   Threefry nor Philox reproduces the ensemble (a SPEC 3.1 contract violation).
Ncol = size(ySeq, 2);
qc = cell(1, Ncol); yc = cell(1, Ncol);
for t = 1:Ncol
    qc{t} = qSeq(t);
    yc{t} = ySeq(:, t);
end
gen = detectGeneratorCell(seed, R, memberParams, qc, yc);
end

function gen = detectGeneratorCell(seed, R, memberParams, qCell, yCell)
candidates = {'Threefry', 'Philox'};
for c = 1:numel(candidates)
    g = candidates{c};
    ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, memberParams{:});
    [members, streams] = oracleBuild(g, seed, R, memberParams);
    orig = RandStream.getGlobalStream;
    restore = onCleanup(@() RandStream.setGlobalStream(orig));
    ok = true;
    for t = 1:numel(yCell)
        q = qCell{t}; y = yCell{t};
        if isempty(q)
            ens.observe_q([]);
        else
            ens.observe_q(q);
        end
        ens.observe_y(y);
        oracleStep(members, streams, q, y);
        if max(abs(ens.motor_output() - oracleMotor(members))) > 1e-9
            ok = false;
            break;
        end
    end
    clear restore;
    if ok
        gen = g;
        return;
    end
end
error('test_ensemble:noGenerator', ...
    ['Ensemble RNG did not match a Threefry or Philox substream ' ...
     '(NumStreams=runs, StreamIndices=k, Seed=seed) as required by SPEC 3.1.']);
end

% ------------------------------------------------------------------------
% Assertion helpers
% ------------------------------------------------------------------------

function assertNear(name, a, b, tol)
%ASSERTNEAR Elementwise closeness with matching sizes; NaN must align exactly.
if ~isequal(size(a), size(b))
    error('test_ensemble:size', 'FAILED %s: size [%s] != [%s]', ...
        name, num2str(size(a)), num2str(size(b)));
end
nanA = isnan(a); nanB = isnan(b);
if ~isequal(nanA, nanB)
    error('test_ensemble:nanPattern', 'FAILED %s: NaN pattern mismatch', name);
end
d = abs(a(~nanA) - b(~nanB));
if ~isempty(d) && (any(~isfinite(d)) || max(d) > tol)
    error('test_ensemble:near', 'FAILED %s (max diff %g, tol %g)', ...
        name, max(d), tol);
end
end

function assertBit(name, a, b)
%ASSERTBIT Bit-identical equality (NaN counts as equal to NaN).
if ~isequaln(a, b)
    d = NaN;
    if isequal(size(a), size(b))
        finite = isfinite(a) & isfinite(b);
        d = max(abs(a(finite) - b(finite)));
    end
    error('test_ensemble:bit', 'FAILED bit-identical: %s (max finite diff %g)', name, d);
end
end

function assertRngUnchanged(name, before)
after = rng;
if ~(isequal(before.Type, after.Type) && isequal(before.Seed, after.Seed) ...
        && isequal(before.State, after.State))
    error('test_ensemble:globalRng', ...
        'FAILED global RNG changed %s (SPEC 3.5)', name);
end
end
