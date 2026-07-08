function test_ensemble_phase2_blindA
%TEST_ENSEMBLE_PHASE2 Blind behavioural tests for RealTimeCOINEnsemble Phase 2.
%
%   Authored strictly against docs/SPEC_ensemble.md Part 10 (Phase 2), NOT
%   against any implementation. Phase 2 adds cross-run context alignment for the
%   six context-indexed ensemble readouts:
%       responsibilities_vector
%       predicted_context_probabilities_vector
%       sampled_context_count
%       stationary_context_probabilities
%       state_given_context_probability
%       state_feedback_given_context_probability
%
%   These tests are chosen (per SPEC section 10.5) so that the cross-run
%   prototype-Hungarian aligner does NOT have to be re-implemented as an oracle.
%   The independent oracle here builds R ordinary RealTimeCOIN members under the
%   same RNG substream contract used by the Phase-1 blind tests (SPEC section 3:
%   a sub-streamable generator with NumStreams = runs, StreamIndices = k,
%   Seed = seed), so the reference members line up bit-for-bit with the
%   ensemble's members. RealTimeCOIN itself is NOT the code under test and is
%   used fully.
%
%   These tests intentionally FAIL against the signature-only Phase-2 stub
%   (which returns NaN / empty); they are the oracle for the real implementation.
%
%   Coverage maps to SPEC section 10.5:
%     (1) runs == 1 reduction to the single member (scalar and MD; vectors and
%         density Maps identical, including novel-slot placement)
%     (2) probability conservation (four probability quantities sum to 1)
%     (3) member-permutation invariance (via equal-weight order independence +
%         reproducibility; see documented limitation below)
%     (4) reproducibility and executor invariance (max_cores, segment_length)
%     (5) trivial-alignment regime: aligned average == naive slot/key average
%     (6) per-context density normalisation (integrates to ~1)
%     (7) shape / key correctness
%
%   Documented spec ambiguities / interpretations:
%     * Generator family (SPEC 3.1 says "e.g. Threefry or Philox"): the substream
%       generator is not uniquely pinned, so the oracle auto-detects which of
%       {Threefry, Philox} reproduces the ensemble (via the Phase-1 motor_output,
%       which is implemented) and then holds every Phase-2 query to exact
%       agreement under that generator. If NEITHER matches, that is a SPEC 3.1
%       contract violation and the test errors -- it does not weaken assertions.
%     * Stream indexing: member k (k = 1..R) uses StreamIndices = k (1-based, as
%       MATLAB requires), NumStreams = runs.
%     * Trivial-alignment regime (SPEC 10.5.5): to make cross-run matching an
%       unambiguous identity WITHOUT re-implementing the aligner, the scenarios
%       are engineered so every member instantiates exactly ONE context (K == 1)
%       under a constant, small perturbation with a low new-context rate. With a
%       single real context per member the matching is forced (the one real
%       context maps to reference label 1, novel maps to the reference novel
%       slot) independent of prototype values, so the aligned average must equal
%       the naive slot-by-slot (vectors) / key-by-key (densities) member average.
%       The precondition K == 1 for every member is asserted, so a scenario that
%       failed to be trivial would surface as an explicit failure, not a silent
%       weakening.
%     * Member-permutation invariance (SPEC 10.5.3): member storage order is not
%       externally observable (members are private and seeded by index), so the
%       test cannot literally reorder them. It is covered two ways: (a) the
%       aligned average equals an equal-weight naive average, which is provably
%       order-independent (asserted by evaluating the naive oracle in original
%       and reversed member order), and (b) reproducibility -- two same-(seed,
%       runs) ensembles give bit-identical Phase-2 outputs. This is the strongest
%       statement obtainable without private access; documented as a limitation.

check_runs_one_reduction_scalar();
check_runs_one_reduction_md();
check_probability_conservation();
check_reproducibility_and_executor();
check_trivial_and_permutation_scalar();
check_trivial_and_permutation_md();
check_density_normalisation_scalar();
check_shape_and_keys();
end

% ========================================================================
% (1) runs == 1 reduction -- scalar model
% ========================================================================
function check_runs_one_reduction_scalar()
mc = 4;
p = {'num_particles', 60, 'max_contexts', mc, 'infer_bias', true};
seed = 55;
[qSeq, ySeq] = ctxScalarStream();
grid = linspace(-1.2, 1.2, 9);
gen = detectGenerator(seed, 1, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, 1, p);
orig = RandStream.getGlobalStream;
guard = onCleanup(@() RandStream.setGlobalStream(orig));

L = mc + 1;
for t = 1:numel(ySeq)
    ens.observe_q(qSeq(t));
    ens.observe_y(ySeq(t));
    oracleStep(members, streams, qSeq(t), ySeq(t));
    m = members{1};

    % Probability vectors: identical to the single member (incl. novel slot).
    for name = {'responsibilities_vector', ...
            'predicted_context_probabilities_vector', 'sampled_context_count'}
        f = name{1};
        pe = ens.(f)();
        pm = m.(f)();
        testutil.assertSize(['runs1 scalar ' f ' length'], pe, [1, L]);
        assertVecEqual(['runs1 scalar ' f], pe, pm, 1e-12);
    end

    % Stationary probabilities: identical (handles [] when K == 0).
    assertVecEqual('runs1 scalar stationary', ...
        ens.stationary_context_probabilities(), ...
        m.stationary_context_probabilities(), 1e-12);

    % Per-context density Maps: identical keys and identical values.
    assertMapClose('runs1 scalar state|ctx', ...
        ens.state_given_context_probability(grid), ...
        m.state_given_context_probability(grid), 1e-12, numel(grid));
    assertMapClose('runs1 scalar fb|ctx', ...
        ens.state_feedback_given_context_probability(grid), ...
        m.state_feedback_given_context_probability(grid), 1e-12, numel(grid));
end
end

% ========================================================================
% (1) runs == 1 reduction -- multi-dimensional model (state_dim == 2)
% ========================================================================
function check_runs_one_reduction_md()
mc = 4;
p = {'state_dim', 2, 'num_particles', 60, 'max_contexts', mc};
seed = 66;
[qSeq, ySeq] = ctxMdStream();
grid = [linspace(-1, 1, 6); linspace(-0.5, 0.5, 6)];
gen = detectGenerator(seed, 1, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', 1, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, 1, p);
orig = RandStream.getGlobalStream;
guard = onCleanup(@() RandStream.setGlobalStream(orig));

L = mc + 1;
for t = 1:size(ySeq, 2)
    ens.observe_q(qSeq(t));
    ens.observe_y(ySeq(:, t));
    oracleStep(members, streams, qSeq(t), ySeq(:, t));
    m = members{1};

    for name = {'responsibilities_vector', ...
            'predicted_context_probabilities_vector', 'sampled_context_count'}
        f = name{1};
        pe = ens.(f)();
        testutil.assertSize(['runs1 MD ' f ' length'], pe, [1, L]);
        assertVecEqual(['runs1 MD ' f], pe, m.(f)(), 1e-12);
    end

    assertVecEqual('runs1 MD stationary', ...
        ens.stationary_context_probabilities(), ...
        m.stationary_context_probabilities(), 1e-12);

    assertMapClose('runs1 MD state|ctx', ...
        ens.state_given_context_probability(grid), ...
        m.state_given_context_probability(grid), 1e-12, size(grid, 2));
    assertMapClose('runs1 MD fb|ctx', ...
        ens.state_feedback_given_context_probability(grid), ...
        m.state_feedback_given_context_probability(grid), 1e-12, size(grid, 2));
end
end

% ========================================================================
% (2) Probability conservation with R > 1 (>= 1 context present)
% ========================================================================
function check_probability_conservation()
mc = 4;
L = mc + 1;
seed = 123;
SUM_TOL = 1e-9;

% --- scalar ---
pS = {'num_particles', 60, 'max_contexts', mc, 'infer_bias', true};
[qS, yS] = ctxScalarStream();
genS = detectGenerator(seed, 5, pS, qS, yS);
ensS = RealTimeCOINEnsemble('runs', 5, 'seed', seed, pS{:});
[memS, strS] = oracleBuild(genS, seed, 5, pS);
origS = RandStream.getGlobalStream;
guardS = onCleanup(@() RandStream.setGlobalStream(origS));
for t = 1:numel(yS)
    ensS.observe_q(qS(t)); ensS.observe_y(yS(t));
    oracleStep(memS, strS, qS(t), yS(t));
end
[~, KrefS] = referenceFrame(memS);
testutil.assertTrue('conservation scalar Kref>=1', KrefS >= 1);
conservationChecks('scalar', ensS, L, KrefS, SUM_TOL);
clear guardS;

% --- MD ---
pM = {'state_dim', 2, 'num_particles', 60, 'max_contexts', mc};
[qM, yM] = ctxMdStream();
genM = detectGenerator(seed, 4, pM, qM, yM);
ensM = RealTimeCOINEnsemble('runs', 4, 'seed', seed, pM{:});
[memM, strM] = oracleBuild(genM, seed, 4, pM);
origM = RandStream.getGlobalStream;
guardM = onCleanup(@() RandStream.setGlobalStream(origM));
for t = 1:size(yM, 2)
    ensM.observe_q(qM(t)); ensM.observe_y(yM(:, t));
    oracleStep(memM, strM, qM(t), yM(:, t));
end
[~, KrefM] = referenceFrame(memM);
testutil.assertTrue('conservation MD Kref>=1', KrefM >= 1);
conservationChecks('MD', ensM, L, KrefM, SUM_TOL);
end

function conservationChecks(tag, ens, L, Kref, tol)
%CONSERVATIONCHECKS Sum-to-1, non-negativity, and length for the four
%   probability quantities (SPEC 10.3 zero-fill rule => exact conservation).
for name = {'responsibilities_vector', ...
        'predicted_context_probabilities_vector', 'sampled_context_count'}
    f = name{1};
    v = ens.(f)();
    testutil.assertSize([tag ' ' f ' length'], v, [1, L]);
    testutil.assertTrue([tag ' ' f ' nonneg'], all(v >= -tol));
    testutil.assertTrue([tag ' ' f ' sums to 1'], abs(sum(v) - 1) <= tol);
end
s = ens.stationary_context_probabilities();
testutil.assertSize([tag ' stationary length'], s, [1, Kref]);
testutil.assertTrue([tag ' stationary nonneg'], all(s >= -tol));
testutil.assertTrue([tag ' stationary sums to 1'], abs(sum(s) - 1) <= tol);
end

% ========================================================================
% (4) Reproducibility and executor invariance (SPEC 10.2/10.5.4)
% ========================================================================
function check_reproducibility_and_executor()
mc = 4;
p = {'num_particles', 60, 'max_contexts', mc, 'infer_bias', true};
seed = 202;
[qSeq, ySeq] = ctxScalarStream();
grid = linspace(-1, 1, 9);

% Reproducibility: identical config => bit-identical Phase-2 outputs.
ensA = RealTimeCOINEnsemble('runs', 5, 'seed', seed, p{:});
ensB = RealTimeCOINEnsemble('runs', 5, 'seed', seed, p{:});
% Executor invariance: serial vs parallel, different segment_length.
ensS = RealTimeCOINEnsemble('runs', 5, 'seed', seed, ...
    'max_cores', 0, 'segment_length', 1, p{:});
ensP = RealTimeCOINEnsemble('runs', 5, 'seed', seed, ...
    'max_cores', 2, 'segment_length', 4, p{:});

for t = 1:numel(ySeq)
    ensA.observe_q(qSeq(t)); ensA.observe_y(ySeq(t));
    ensB.observe_q(qSeq(t)); ensB.observe_y(ySeq(t));
    ensS.observe_q(qSeq(t)); ensS.observe_y(ySeq(t));
    ensP.observe_q(qSeq(t)); ensP.observe_y(ySeq(t));

    assertPhase2Bit('reproducible', ensA, ensB, grid);
    assertPhase2Bit('executor', ensS, ensP, grid);
end
end

function assertPhase2Bit(tag, a, b, grid)
%ASSERTPHASE2BIT Bit-identical Phase-2 outputs between two ensembles.
for name = {'responsibilities_vector', ...
        'predicted_context_probabilities_vector', 'sampled_context_count'}
    f = name{1};
    assertBit([tag ' ' f], a.(f)(), b.(f)());
end
assertBit([tag ' stationary'], ...
    a.stationary_context_probabilities(), b.stationary_context_probabilities());
assertMapBit([tag ' state|ctx'], ...
    a.state_given_context_probability(grid), b.state_given_context_probability(grid));
assertMapBit([tag ' fb|ctx'], ...
    a.state_feedback_given_context_probability(grid), ...
    b.state_feedback_given_context_probability(grid));
end

% ========================================================================
% (5)+(3) Trivial-alignment regime + permutation invariance -- scalar
% ========================================================================
function check_trivial_and_permutation_scalar()
mc = 4;
p = {'num_particles', 100, 'max_contexts', mc, 'gamma_context', 1e-4};
seed = 314;
[qSeq, ySeq] = trivialScalarStream();
grid = linspace(-1.0, 1.0, 11);
gen = detectGenerator(seed, 4, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', 4, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, 4, p);
orig = RandStream.getGlobalStream;
guard = onCleanup(@() RandStream.setGlobalStream(orig));
for t = 1:numel(ySeq)
    ens.observe_q(qSeq(t)); ens.observe_y(ySeq(t));
    oracleStep(members, streams, qSeq(t), ySeq(t));
end

assertTrivialFrame('trivial scalar', members);   % every member has K == 1

% Vectors: aligned average == naive slot-by-slot member average, and the
% naive average is order-independent (permutation invariance).
for name = {'responsibilities_vector', ...
        'predicted_context_probabilities_vector', 'sampled_context_count'}
    f = name{1};
    naive = naiveVectorAverage(members, f);
    naiveRev = naiveVectorAverage(members(end:-1:1), f);
    assertVecEqual(['perm-order ' f], naive, naiveRev, 1e-12);
    assertVecEqual(['trivial scalar ' f], ens.(f)(), naive, 1e-9);
end

% Stationary: aligned == naive average of member stationary vectors.
naiveStat = naiveVectorAverage(members, 'stationary_context_probabilities');
assertVecEqual('trivial scalar stationary', ...
    ens.stationary_context_probabilities(), naiveStat, 1e-9);

% Densities: aligned key-by-key == naive key-by-key member average.
assertMapClose('trivial scalar state|ctx', ...
    ens.state_given_context_probability(grid), ...
    naiveDensityAverage(members, 'state_given_context_probability', grid), ...
    1e-9, numel(grid));
assertMapClose('trivial scalar fb|ctx', ...
    ens.state_feedback_given_context_probability(grid), ...
    naiveDensityAverage(members, 'state_feedback_given_context_probability', grid), ...
    1e-9, numel(grid));
end

% ========================================================================
% (5)+(6) Trivial-alignment regime + MD density normalisation -- MD model
% ========================================================================
function check_trivial_and_permutation_md()
mc = 4;
p = {'state_dim', 2, 'num_particles', 100, 'max_contexts', mc, 'gamma_context', 1e-4};
seed = 271;
[qSeq, ySeq] = trivialMdStream();
grid = [linspace(-0.6, 0.6, 7); linspace(-0.6, 0.6, 7)];
gen = detectGenerator(seed, 4, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', 4, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, 4, p);
orig = RandStream.getGlobalStream;
guard = onCleanup(@() RandStream.setGlobalStream(orig));
for t = 1:size(ySeq, 2)
    ens.observe_q(qSeq(t)); ens.observe_y(ySeq(:, t));
    oracleStep(members, streams, qSeq(t), ySeq(:, t));
end

assertTrivialFrame('trivial MD', members);

for name = {'responsibilities_vector', ...
        'predicted_context_probabilities_vector', 'sampled_context_count'}
    f = name{1};
    naive = naiveVectorAverage(members, f);
    assertVecEqual(['trivial MD ' f], ens.(f)(), naive, 1e-9);
end
assertVecEqual('trivial MD stationary', ...
    ens.stationary_context_probabilities(), ...
    naiveVectorAverage(members, 'stationary_context_probabilities'), 1e-9);

assertMapClose('trivial MD state|ctx', ...
    ens.state_given_context_probability(grid), ...
    naiveDensityAverage(members, 'state_given_context_probability', grid), ...
    1e-9, size(grid, 2));

% (6) MD per-context density normalisation: integrate each aligned context
% density to ~1. Center integration at the reference member's context
% prototype (K == 1 => reference label 1 maps to member context 1); the
% prototype only positions the integration window, it is not the oracle answer.
[refIdx, Kref] = referenceFrame(members);
testutil.assertTrue('trivial MD Kref==1', Kref == 1);
refAlign = members{refIdx}.context_alignment();
center = refAlign.global_contexts.state_mean(:, 1);
[~, pooledCov] = ens.state_moments();
halfStd = sqrt(max(diag(pooledCov), 1e-6));
dmap = ens.state_given_context_probability(grid);
ks = cell2mat(dmap.keys);
for j = 1:numel(ks)
    key = ks(j);
    densFun = @(X) mapDensity(ens.state_given_context_probability(X), key);
    [integ, vals] = testutil.integrate2d(densFun, center, halfStd);
    testutil.assertTrue(sprintf('trivial MD ctx %d density nonneg', key), ...
        all(vals >= -1e-12));
    testutil.assertTrue(sprintf('trivial MD ctx %d integrates ~1 (got %.4f)', key, integ), ...
        abs(integ - 1) < 0.1);
end
end

% ========================================================================
% (6) Per-context density normalisation -- scalar (wide fine grid, trapz)
% ========================================================================
function check_density_normalisation_scalar()
mc = 4;
p = {'num_particles', 80, 'max_contexts', mc, 'infer_bias', true};
seed = 909;
[qSeq, ySeq] = ctxScalarStream();
gen = detectGenerator(seed, 4, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', 4, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, 4, p);
orig = RandStream.getGlobalStream;
guard = onCleanup(@() RandStream.setGlobalStream(orig));
for t = 1:numel(ySeq)
    ens.observe_q(qSeq(t)); ens.observe_y(ySeq(t));
    oracleStep(members, streams, qSeq(t), ySeq(t));
end

% Wide, fine grid: spacing 1e-3 resolves the small (~1e-1) density widths and
% spans well beyond any context mean.
grid = linspace(-4, 4, 8001);
for method = {'state_given_context_probability', ...
        'state_feedback_given_context_probability'}
    f = method{1};
    dmap = ens.(f)(grid);
    testutil.assertTrue(['density-norm scalar ' f ' is Map'], isa(dmap, 'containers.Map'));
    ks = cell2mat(dmap.keys);
    testutil.assertTrue(['density-norm scalar ' f ' has a context'], ~isempty(ks));
    for j = 1:numel(ks)
        row = dmap(ks(j));
        testutil.assertSize(['density-norm scalar ' f ' row'], row, [1, numel(grid)]);
        testutil.assertTrue(['density-norm scalar ' f ' nonneg'], all(row >= -1e-12));
        integ = trapz(grid, row);
        testutil.assertTrue( ...
            sprintf('density-norm scalar %s ctx %d integrates ~1 (got %.4f)', f, ks(j), integ), ...
            abs(integ - 1) < 0.05);
    end
end
end

% ========================================================================
% (7) Shape / key correctness (SPEC 10.5.7)
% ========================================================================
function check_shape_and_keys()
mc = 5;
L = mc + 1;
p = {'num_particles', 60, 'max_contexts', mc, 'infer_bias', true};
seed = 4242;
[qSeq, ySeq] = ctxScalarStream();
grid = linspace(-1.5, 1.5, 13);
gen = detectGenerator(seed, 4, p, qSeq, ySeq);

ens = RealTimeCOINEnsemble('runs', 4, 'seed', seed, p{:});
[members, streams] = oracleBuild(gen, seed, 4, p);
orig = RandStream.getGlobalStream;
guard = onCleanup(@() RandStream.setGlobalStream(orig));
for t = 1:numel(ySeq)
    ens.observe_q(qSeq(t)); ens.observe_y(ySeq(t));
    oracleStep(members, streams, qSeq(t), ySeq(t));
end
[~, Kref] = referenceFrame(members);

% Vector lengths.
for name = {'responsibilities_vector', ...
        'predicted_context_probabilities_vector', 'sampled_context_count'}
    f = name{1};
    testutil.assertSize(['shape ' f], ens.(f)(), [1, L]);
end

% Stationary length: 1xKref (or [] when Kref == 0).
s = ens.stationary_context_probabilities();
if Kref == 0
    testutil.assertTrue('shape stationary empty', isempty(s));
else
    testutil.assertSize('shape stationary', s, [1, Kref]);
end

% Density Map keys subset of 1..Kref; each value a 1xK row.
for method = {'state_given_context_probability', ...
        'state_feedback_given_context_probability'}
    f = method{1};
    dmap = ens.(f)(grid);
    testutil.assertTrue(['shape ' f ' is Map'], isa(dmap, 'containers.Map'));
    ks = cell2mat(dmap.keys);
    testutil.assertTrue(['shape ' f ' keys subset 1..Kref'], ...
        isempty(ks) || (all(ks >= 1) && all(ks <= Kref) && all(ks == round(ks))));
    for j = 1:numel(ks)
        testutil.assertSize(['shape ' f ' row'], dmap(ks(j)), [1, numel(grid)]);
    end
end
end

% ========================================================================
% Observation streams
% ========================================================================
function [qSeq, ySeq] = ctxScalarStream()
%CTXSCALARSTREAM Alternating blocks that instantiate >= 1 context.
qSeq = [1 1 1 2 2 2 1 1];
ySeq = [0.30 0.28 0.32 -0.30 -0.28 -0.31 0.29 0.30];
end

function [qSeq, ySeq] = ctxMdStream()
qSeq = [1 1 1 2 2 2 1 1];
ySeq = [0.30 0.28 0.32 -0.30 -0.28 -0.31 0.29 0.30; ...
        0.20 0.18 0.22 -0.20 -0.18 -0.21 0.19 0.20];
end

function [qSeq, ySeq] = trivialScalarStream()
%TRIVIALSCALARSTREAM Constant small perturbation => a single context (K == 1).
qSeq = [1 1 1 1 1];
ySeq = [0.05 0.05 0.05 0.05 0.05];
end

function [qSeq, ySeq] = trivialMdStream()
qSeq = [1 1 1 1 1];
ySeq = [0.05 0.05 0.05 0.05 0.05; ...
        0.03 0.03 0.03 0.03 0.03];
end

% ========================================================================
% Oracle: RNG substream contract (SPEC section 3) + reference frame
% ========================================================================
function [members, streams] = oracleBuild(gen, seed, R, memberParams)
%ORACLEBUILD Construct R RealTimeCOIN members, each under its own substream.
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
    members{k}.observe_q(q);
    members{k}.observe_y(y);
end
end

function [refIdx, Kref, Kvec] = referenceFrame(members)
%REFERENCEFRAME Reference member = most contexts (ties -> lowest index).
R = numel(members);
Kvec = zeros(1, R);
for k = 1:R
    a = members{k}.context_alignment();
    Kvec(k) = a.K;
end
[Kref, refIdx] = max(Kvec);   % max returns the first index on ties.
end

function assertTrivialFrame(name, members)
%ASSERTTRIVIALFRAME Precondition for the trivial-alignment regime: every member
%   instantiated exactly one context, so cross-run matching is a forced
%   identity and aligned average == naive average (SPEC 10.5.5).
[~, ~, Kvec] = referenceFrame(members);
testutil.assertTrue([name ': every member has K == 1'], all(Kvec == 1));
end

% ========================================================================
% Naive (identity-frame) averaging oracles -- valid ONLY when the frame is
% trivial (all members share the same K == 1 frame). See assertTrivialFrame.
% ========================================================================
function v = naiveVectorAverage(members, method)
%NAIVEVECTORAVERAGE Equal-weight slot-by-slot mean of member vectors.
%   For probability vectors (SPEC 10.3 zero-fill), in a shared identity frame no
%   run lacks a reference context and slots beyond K are already zero, so the
%   aligned average reduces to the plain element-wise mean. Handles the empty
%   ([] stationary when K == 0) case.
R = numel(members);
first = members{1}.(method)();
if isempty(first)
    v = [];
    return;
end
acc = zeros(size(first));
for k = 1:R
    acc = acc + members{k}.(method)();
end
v = acc ./ R;
end

function out = naiveDensityAverage(members, method, values)
%NAIVEDENSITYAVERAGE Key-by-key NaN-omit mean of member density Maps.
%   Valid in a shared identity frame (member local labels equal reference
%   labels). For each reference key, average over the members that HAVE it
%   (SPEC 10.3 NaN-omit rule).
R = numel(members);
maps = cell(1, R);
allKeys = [];
for k = 1:R
    maps{k} = members{k}.(method)(values);
    allKeys = union(allKeys, cell2mat(maps{k}.keys));
end
out = containers.Map('KeyType', 'double', 'ValueType', 'any');
for key = allKeys(:)'
    contrib = [];
    for k = 1:R
        if isKey(maps{k}, key)
            contrib = [contrib; maps{k}(key)]; %#ok<AGROW>
        end
    end
    out(key) = nanOmitColumnMean(contrib);
end
end

function m = nanOmitColumnMean(X)
%NANOMITCOLUMNMEAN Column-wise mean over finite rows; NaN if a column is all
%   non-finite (SPEC NaN-omit-with-all-NaN => NaN).
m = nan(1, size(X, 2));
for j = 1:size(X, 2)
    col = X(:, j);
    fin = isfinite(col);
    if any(fin)
        m(j) = mean(col(fin));
    end
end
end

% ========================================================================
% Generator detection (SPEC 3.1: Threefry or Philox permitted)
% ========================================================================
function gen = detectGenerator(seed, R, memberParams, qSeq, ySeq)
%DETECTGENERATOR Identify which spec-permitted substream generator the ensemble
%   uses, via Phase-1 motor_output agreement over the stream.
Ncol = size(ySeq, 2);
candidates = {'Threefry', 'Philox'};
for c = 1:numel(candidates)
    g = candidates{c};
    ens = RealTimeCOINEnsemble('runs', R, 'seed', seed, memberParams{:});
    [members, streams] = oracleBuild(g, seed, R, memberParams);
    orig = RandStream.getGlobalStream;
    restore = onCleanup(@() RandStream.setGlobalStream(orig));
    ok = true;
    for t = 1:Ncol
        q = qSeq(t);
        y = ySeq(:, t);
        ens.observe_q(q);
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
error('test_ensemble_phase2:noGenerator', ...
    ['Ensemble RNG did not match a Threefry or Philox substream ' ...
     '(NumStreams=runs, StreamIndices=k, Seed=seed) as required by SPEC 3.1.']);
end

function u = oracleMotor(members)
%ORACLEMOTOR Equal-weight mean of per-member motor_output (Phase-1 oracle).
R = numel(members);
first = members{1}.motor_output();
X = zeros(numel(first), R);
X(:, 1) = first;
for k = 2:R
    X(:, k) = members{k}.motor_output();
end
u = mean(X, 2);
end

% ========================================================================
% Assertion helpers
% ========================================================================
function assertVecEqual(name, a, b, tol)
%ASSERTVECEQUAL Closeness with matching sizes; NaN aligns exactly; handles [].
if isempty(a) || isempty(b)
    if ~(isempty(a) && isempty(b))
        error('test_ensemble_phase2:size', ...
            'FAILED %s: one operand empty, the other not', name);
    end
    return;
end
if ~isequal(size(a), size(b))
    error('test_ensemble_phase2:size', 'FAILED %s: size [%s] != [%s]', ...
        name, num2str(size(a)), num2str(size(b)));
end
nanA = isnan(a); nanB = isnan(b);
if ~isequal(nanA, nanB)
    error('test_ensemble_phase2:nanPattern', 'FAILED %s: NaN pattern mismatch', name);
end
d = abs(a(~nanA) - b(~nanB));
if ~isempty(d) && (any(~isfinite(d)) || max(d) > tol)
    error('test_ensemble_phase2:near', 'FAILED %s (max diff %g, tol %g)', ...
        name, max(d), tol);
end
end

function assertBit(name, a, b)
%ASSERTBIT Bit-identical equality (NaN counts as equal to NaN).
if ~isequaln(a, b)
    d = NaN;
    if isequal(size(a), size(b)) && isnumeric(a) && isnumeric(b)
        finite = isfinite(a) & isfinite(b);
        if any(finite(:))
            d = max(abs(a(finite) - b(finite)));
        end
    end
    error('test_ensemble_phase2:bit', ...
        'FAILED bit-identical: %s (max finite diff %g)', name, d);
end
end

function assertMapClose(name, ma, mb, tol, K)
%ASSERTMAPCLOSE Same keys and elementwise-close values; each value 1xK.
if ~isa(ma, 'containers.Map')
    error('test_ensemble_phase2:map', 'FAILED %s: not a containers.Map', name);
end
ka = sort(cell2mat(ma.keys));
kb = sort(cell2mat(mb.keys));
if ~isequal(ka, kb)
    error('test_ensemble_phase2:mapKeys', 'FAILED %s: key sets differ [%s] vs [%s]', ...
        name, num2str(ka), num2str(kb));
end
for key = ka(:)'
    va = ma(key); vb = mb(key);
    testutil.assertSize(sprintf('%s value row (key %d)', name, key), va, [1, K]);
    assertVecEqual(sprintf('%s value (key %d)', name, key), va, vb, tol);
end
end

function assertMapBit(name, ma, mb)
%ASSERTMAPBIT Same keys and bit-identical values.
ka = sort(cell2mat(ma.keys));
kb = sort(cell2mat(mb.keys));
if ~isequal(ka, kb)
    error('test_ensemble_phase2:mapKeys', 'FAILED %s: key sets differ [%s] vs [%s]', ...
        name, num2str(ka), num2str(kb));
end
for key = ka(:)'
    assertBit(sprintf('%s (key %d)', name, key), ma(key), mb(key));
end
end

function v = mapDensity(m, key)
%MAPDENSITY Fetch a density row from a Map (helper for anonymous handles).
v = m(key);
end
