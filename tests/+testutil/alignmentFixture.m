function coin = alignmentFixture(includeNonModal)
%TESTUTIL.ALIGNMENTFIXTURE Two-context particle fixture for alignment tests.
%   coin = testutil.alignmentFixture(includeNonModal) builds a RealTimeCOIN whose
%   particles encode a known two-context configuration with a deterministic
%   local-to-global label permutation, so global context alignment has a
%   verifiable ground truth. When includeNonModal is true an extra particle with
%   a single, off-prototype context is appended to exercise modal-subset logic.
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

coin = testutil.loadFixtureModel(template, D);
end
