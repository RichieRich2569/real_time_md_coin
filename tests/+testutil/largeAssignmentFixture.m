function coin = largeAssignmentFixture()
%TESTUTIL.LARGEASSIGNMENTFIXTURE Ten-context particle fixture for alignment tests.
%   coin = testutil.largeAssignmentFixture() builds a two-particle RealTimeCOIN in
%   which one particle labels ten contexts in canonical order and the other in
%   reverse order. Alignment must therefore recover the known reverse permutation,
%   stressing the label-matching logic at the maximum context count.
P = 2;
maxContexts = 10;
Cmax = maxContexts + 1;
template = RealTimeCOIN('num_particles', P, 'max_contexts', maxContexts);
diagState = template.diagnostics();
D = diagState.raw;

D.C = maxContexts * ones(1, P);
D.context = ones(1, P);
D.previous_context = D.context;
D.Q = 1;
D.n_cue = zeros(Cmax, 1, P);
D.global_cue_probabilities = ones(1, P);
D.state_filtered_mean = zeros(Cmax, P);
D.state_filtered_var = 0.01 * ones(Cmax, P);
D.state_mean = D.state_filtered_mean;
D.state_var = D.state_filtered_var;
D.state_feedback_mean = D.state_filtered_mean;
D.state_feedback_var = D.state_filtered_var;
D.retention = zeros(Cmax, P);
D.drift = zeros(Cmax, P);
D.bias = zeros(Cmax, P);
D.bias_mean = zeros(Cmax, P);
D.bias_var = ones(Cmax, P);
D.dynamics_mean = zeros(2, Cmax, P);
D.dynamics_covar = repmat(eye(2), 1, 1, Cmax, P);
D.local_cue_matrix = ones(Cmax, 1, P);
D.local_transition_matrix = zeros(Cmax, Cmax, P);
D.global_transition_probabilities = zeros(Cmax, P);
D.predicted_probabilities = zeros(Cmax, P);
D.responsibilities = zeros(Cmax, P);
D.prior_probabilities = zeros(Cmax, P);
D.probability_cue = ones(Cmax, P);
D.probability_state_feedback = ones(Cmax, P);

for p = 1:P
    if p == 1
        localToGlobal = 1:10;
    else
        localToGlobal = 10:-1:1;
    end
    for local = 1:10
        globalIdx = localToGlobal(local);
        D.state_filtered_mean(local, p) = globalIdx;
        D.state_mean(local, p) = globalIdx;
        D.state_feedback_mean(local, p) = globalIdx;
        D.retention(local, p) = 0.5 + 0.01 * globalIdx;
        D.drift(local, p) = 0.1 * globalIdx;
        D.dynamics_mean(:, local, p) = [D.retention(local, p); D.drift(local, p)];
        D.local_transition_matrix(local, local, p) = 1;
        D.global_transition_probabilities(local, p) = 0.1;
        D.predicted_probabilities(local, p) = 0.1;
        D.responsibilities(local, p) = 0.1;
        D.prior_probabilities(local, p) = 0.1;
    end
end

coin = testutil.loadFixtureModel(template, D);
end
