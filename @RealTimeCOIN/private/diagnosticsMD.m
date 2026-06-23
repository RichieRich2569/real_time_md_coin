function S = diagnosticsMD(obj)
%DIAGNOSTICSMD Multi-dimensional diagnostics summary.
%
%   Multi-dimensional counterpart of diagnostics.m. Returns globally aligned,
%   per-context parameter estimates and context probabilities for the
%   N-dimensional model. Per-context dynamics are reported as the augmented
%   matrix Theta = [A | d] split into the retention matrix A (N x N x K) and
%   the drift vector d (N x K), alongside the filtered state mean/covariance
%   and the (optional) observation bias. Context probabilities reuse the
%   dimension-agnostic global-weight machinery.
%
%   Fields (K = number of globally aligned contexts):
%       trial, K
%       A                : N x N x K   retention matrices per context
%       drift            : N x K       drift vectors per context
%       bias             : N x K       observation bias per context
%       state_mean       : N x K       filtered state mean per context
%       state_cov        : N x N x K   filtered state covariance per context
%       predicted_probabilities, responsibilities : K x 1 mean context weights
%       predicted_probabilities_particles, responsibilities_particles :
%                          Cmax x nModal raw aligned weights
%       context          : 1 x nModal  sampled global context per modal particle
%       transition_prob  : K x (K+1)   global transition prototype
%       cue_prob         : K x Q       global cue prototype
%       alignment, raw

    N = obj.state_dim;
    alignment = obj.ensureContextAlignment();
    proto = alignment.global_contexts;
    Km = alignment.K;

    S = struct();
    S.trial = obj.trial;
    S.K = Km;

    % Per-context dynamics: split Theta = [A | d].
    S.A = proto.theta_mean(:, 1:N, :);
    S.drift = reshape(proto.theta_mean(:, N+1, :), N, Km);
    S.bias = proto.bias_mean;
    S.state_mean = proto.state_mean;
    S.state_cov = proto.state_cov;

    % Context probabilities (aligned across modal particles).
    Wpred = obj.globalContextWeights(obj.D.predicted_probabilities, alignment);
    Wresp = obj.globalContextWeights(obj.D.responsibilities, alignment);
    S.predicted_probabilities_particles = Wpred;
    S.responsibilities_particles = Wresp;
    S.predicted_probabilities = mean(Wpred(1:Km, :), 2);
    S.responsibilities = mean(Wresp(1:Km, :), 2);

    S.context = obj.globalSampledContexts(alignment);
    S.transition_prob = proto.transition_prob;
    S.cue_prob = proto.cue_prob;

    S.alignment = alignment;
    S.raw = obj.D;
end
