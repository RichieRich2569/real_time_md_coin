function test_global_alignment
%TEST_GLOBAL_ALIGNMENT Lazy global context alignment for context-facing APIs.

rng(6);

coin = alignmentFixture(false);
alignment = coin.context_alignment();

assert(alignment.K == 2, 'Alignment should use the modal cardinality');
assert(all(alignment.modal_particle_mask), 'All fixture particles should be modal');
assert(isequal(alignment.assignment(1:2,1), [1;2]), 'Anchor particle should keep canonical labels');
assert(isequal(alignment.assignment(1:2,2), [2;1]), 'Swapped particle should map back to global labels');
assert(abs(alignment.global_contexts.state_mean(1) + 1) < 1e-6, 'Global context 1 prototype mismatch');
assert(abs(alignment.global_contexts.state_mean(2) - 1) < 1e-6, 'Global context 2 prototype mismatch');

resp = coin.context_probabilities();
assert(abs(resp(1) - 0.6) < 1e-12, 'Global responsibility for context 1 mismatch');
assert(abs(resp(2) - 0.3) < 1e-12, 'Global responsibility for context 2 mismatch');
assert(abs(resp(3) - 0.1) < 1e-12, 'Novel responsibility bucket mismatch');

pred = coin.predicted_context_probabilities();
assert(max(abs(pred(1:3) - [0.7 0.2 0.1])) < 1e-12, 'Global predicted probabilities mismatch');

coinModal = alignmentFixture(true);
alignmentModal = coinModal.context_alignment();
assert(alignmentModal.K == 2, 'Modal subset should keep the two-context cardinality');
assert(sum(alignmentModal.modal_particle_mask) == 3, 'Exactly three particles should be in the modal subset');
respModal = coinModal.context_probabilities();
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
end

function coin = alignmentFixture(includeNonModal)
P = 3 + double(includeNonModal);
maxContexts = 3;
Cmax = maxContexts + 1;
template = RealTimeCOIN('num_particles', P, 'max_contexts', maxContexts);
diagState = template.diagnostics();
D = diagState.raw;

D.C = 2 * ones(1, P);
D.context = [1 2 1 ones(1, P-3)];
D.previous_context = D.context;
D.Q = 1;
D.n_cue = zeros(Cmax, 2, P);
D.global_cue_probabilities = 0.5 * ones(2, P);

D.state_filtered_mean = zeros(Cmax, P);
D.state_filtered_var = ones(Cmax, P);
D.state_mean = zeros(Cmax, P);
D.state_var = ones(Cmax, P);
D.state_feedback_mean = zeros(Cmax, P);
D.state_feedback_var = ones(Cmax, P);
D.retention = zeros(Cmax, P);
D.drift = zeros(Cmax, P);
D.bias = zeros(Cmax, P);
D.bias_mean = zeros(Cmax, P);
D.bias_var = zeros(Cmax, P);
D.dynamics_mean = zeros(2, Cmax, P);
D.dynamics_covar = zeros(2, 2, Cmax, P);
D.local_cue_matrix = zeros(Cmax, 2, P);
D.local_transition_matrix = zeros(Cmax, Cmax, P);
D.global_transition_probabilities = zeros(Cmax, P);
D.predicted_probabilities = zeros(Cmax, P);
D.responsibilities = zeros(Cmax, P);
D.prior_probabilities = zeros(Cmax, P);
D.probability_cue = ones(Cmax, P);
D.probability_state_feedback = ones(Cmax, P);

globalStateMean = [-1 1];
globalRetention = [0.8 0.6];
globalDrift = [-0.1 0.1];
globalCue = [0.9 0.1; 0.1 0.9];
globalTransition = [0.8 0.1 0.1; 0.2 0.7 0.1];
globalBeta = [0.55 0.35 0.1];
globalPred = [0.7 0.2 0.1];
globalResp = [0.6 0.3 0.1];

for p = 1:3
    if p == 2
        localToGlobal = [2 1];
    else
        localToGlobal = [1 2];
    end

    for local = 1:2
        globalIdx = localToGlobal(local);
        D.state_filtered_mean(local,p) = globalStateMean(globalIdx);
        D.state_filtered_var(local,p) = 0.02;
        D.state_mean(local,p) = globalStateMean(globalIdx);
        D.state_var(local,p) = 0.02;
        D.state_feedback_mean(local,p) = globalStateMean(globalIdx);
        D.state_feedback_var(local,p) = 0.02;
        D.retention(local,p) = globalRetention(globalIdx);
        D.drift(local,p) = globalDrift(globalIdx);
        D.dynamics_mean(:,local,p) = [globalRetention(globalIdx); globalDrift(globalIdx)];
        D.dynamics_covar(:,:,local,p) = diag([0.01 0.01]);
        D.local_cue_matrix(local,:,p) = globalCue(globalIdx,:);
        D.global_transition_probabilities(local,p) = globalBeta(globalIdx);
        D.predicted_probabilities(local,p) = globalPred(globalIdx);
        D.responsibilities(local,p) = globalResp(globalIdx);
        D.prior_probabilities(local,p) = globalPred(globalIdx);
    end

    for local = 1:2
        globalFrom = localToGlobal(local);
        for destLocal = 1:2
            globalTo = localToGlobal(destLocal);
            D.local_transition_matrix(local,destLocal,p) = globalTransition(globalFrom, globalTo);
        end
        D.local_transition_matrix(local,3,p) = globalTransition(globalFrom, 3);
    end

    D.local_cue_matrix(3,:,p) = [0.5 0.5];
    D.local_transition_matrix(3,3,p) = 1;
    D.global_transition_probabilities(3,p) = globalBeta(3);
    D.predicted_probabilities(3,p) = globalPred(3);
    D.responsibilities(3,p) = globalResp(3);
    D.prior_probabilities(3,p) = globalPred(3);
end

if includeNonModal
    p = 4;
    D.C(p) = 1;
    D.context(p) = 1;
    D.previous_context(p) = 1;
    D.state_filtered_mean(1,p) = 10;
    D.state_filtered_var(1,p) = 0.02;
    D.state_mean(1,p) = 10;
    D.state_var(1,p) = 0.02;
    D.state_feedback_mean(1,p) = 10;
    D.state_feedback_var(1,p) = 0.02;
    D.retention(1,p) = 0.1;
    D.drift(1,p) = 10;
    D.dynamics_mean(:,1,p) = [0.1; 10];
    D.dynamics_covar(:,:,1,p) = diag([0.01 0.01]);
    D.local_cue_matrix(1,:,p) = [0.5 0.5];
    D.local_transition_matrix(1,1,p) = 1;
    D.global_transition_probabilities(1,p) = 1;
    D.predicted_probabilities(1,p) = 1;
    D.responsibilities(1,p) = 1;
    D.prior_probabilities(1,p) = 1;
end

coin = loadFixtureModel(template, D);
end

function coin = loadFixtureModel(template, D)
model = struct();
model.properties = struct();
props = properties(template);
for i = 1:numel(props)
    if ~strcmp(props{i}, 'Trial')
        model.properties.(props{i}) = template.(props{i});
    end
end
model.D = D;
model.pending_q = [];
model.trial = 1;
model.cue_values = 1;
model.state_version = 1;

tmpfile = [tempname, '.mat'];
save(tmpfile, 'model');
coin = RealTimeCOIN('num_particles', 1);
coin.loadModel(tmpfile);
delete(tmpfile);
end
