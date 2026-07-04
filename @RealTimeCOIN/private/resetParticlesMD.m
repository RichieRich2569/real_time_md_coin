function resetParticlesMD(obj)
%RESETPARTICLESMD Initialise particle storage for the N-dimensional model.
%
%   Multi-dimensional counterpart of resetParticles.m. Latent state is now an
%   N-vector per context per particle, with full N-by-N covariances, and the
%   dynamics parameters are an augmented matrix Theta = [A | d] drawn from the
%   matrix-normal prior (see dynamicsPriorMD.m). Context-inference fields
%   (counts, transition/cue matrices, context responsibilities) keep the same
%   shapes and semantics as the scalar model and are reused unchanged by the
%   shared context-prediction routines.
%
%   Tensor layout (N = state_dim, Cmax = max_contexts + 1, P = num_particles):
%       state_filtered_mean / state_mean / state_feedback_mean : N x Cmax x P
%       state_filtered_cov  / state_cov  / state_feedback_cov  : N x N x Cmax x P
%       Theta            : N x (N+1) x Cmax x P   (A = Theta(:,1:N), d = (:,N+1))
%       bias             : N x Cmax x P
%       Lambda_xx        : (N+1) x (N+1) x Cmax x P
%       Lambda_yx        : N x (N+1) x Cmax x P
%       bias_info_ss     : N x Cmax x P
%       bias_precision_ss: N x N x Cmax x P

    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;

    D = struct();
    D.C = ones(1, P);
    D.context = ones(1, P);
    D.previous_context = ones(1, P);
    D.Q = 0;

    % --- Context-inference sufficient statistics (shared with scalar model) ---
    D.n_context = zeros(Cmax, Cmax, P);
    D.n_cue = zeros(Cmax, 1, P);
    D.global_transition_probabilities = zeros(Cmax, P);
    D.global_transition_probabilities(1,:) = 1;
    D.global_cue_probabilities = ones(1, P);

    % --- Dynamics parameter sufficient statistics (matrix accumulators) ---
    D.Lambda_xx = zeros(N+1, N+1, Cmax, P);
    D.Lambda_yx = zeros(N, N+1, Cmax, P);
    D.bias_info_ss = zeros(N, Cmax, P);
    D.bias_precision_ss = zeros(N, N, Cmax, P);

    % --- Sample dynamics parameters Theta = [A | d] from the matrix-normal prior ---
    [M0, ~, V0] = obj.dynamicsPriorMD();
    Q = obj.processNoiseCov();
    D.Theta = zeros(N, N+1, Cmax, P);
    for p = 1:P
        for c = 1:Cmax
            D.Theta(:,:,c,p) = obj.sampleStableTheta(M0, Q, V0);
        end
    end

    if obj.infer_bias
        biasVar = obj.precisionToVariance(obj.prior_precision_bias);
        D.bias = obj.prior_mean_bias + sqrt(biasVar) * randn(N, Cmax, P);
    else
        D.bias = zeros(N, Cmax, P);
    end

    % --- Seed every context to its stationary state distribution ---
    D.state_filtered_mean = zeros(N, Cmax, P);
    D.state_filtered_cov = zeros(N, N, Cmax, P);
    for p = 1:P
        for c = 1:Cmax
            A = D.Theta(:,1:N,c,p);
            d = D.Theta(:,N+1,c,p);
            D.state_filtered_mean(:,c,p) = obj.stationaryStateMeanMD(A, d);
            D.state_filtered_cov(:,:,c,p) = obj.stationaryStateCovMD(A, Q);
        end
    end

    D.previous_state_filtered_mean = D.state_filtered_mean;
    D.previous_state_filtered_cov = D.state_filtered_cov;
    D.state_mean = D.state_filtered_mean;
    D.state_cov = D.state_filtered_cov;

    R = obj.observationNoiseCov();
    D.state_feedback_mean = D.state_mean + D.bias;
    D.state_feedback_cov = D.state_cov + repmat(R, 1, 1, Cmax, P);

    % --- Context probabilities (shared shapes/semantics with scalar model) ---
    D.prior_probabilities = zeros(Cmax, P);
    D.prior_probabilities(1,:) = 1;
    D.predicted_probabilities = D.prior_probabilities;
    D.responsibilities = D.prior_probabilities;
    D.probability_cue = ones(Cmax, P);
    D.probability_state_feedback = ones(Cmax, P);
    D.i_resampled = 1:P;

    obj.D = D; %#ok<*PROP>
    obj.updateLocalTransitionMatrix();
    obj.updateLocalCueMatrix();
    obj.alignment_seed = [];
    obj.invalidateContextAlignment();
end
