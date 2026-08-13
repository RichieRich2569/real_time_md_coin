function [W, M, V] = previewPredictiveFeedback(obj, q)
%PREVIEWPREDICTIVEFEEDBACK One-step predictive feedback mixture (scalar model).
%   [W, M, V] = previewPredictiveFeedback(obj, q) performs a read-only one-step
%   look-ahead from the current posterior, returning the per-particle,
%   per-context Gaussian-mixture components of the next observation given the
%   optional upcoming cue label q (q = [] marginalises the cue out):
%       W : (max_contexts+1)-by-P normalised mixture weights,
%       M : (max_contexts+1)-by-P predictive feedback means,
%       V : (max_contexts+1)-by-P predictive feedback variances.
%   Each component mirrors the scalar Kalman one-step prediction in
%   predictStateFeedback: state s' = a*s + d with variance a^2*var + Q, then
%   feedback mean s' + bias and variance + observation variance. The first
%   not-yet-instantiated ("novel") context uses the stationary prediction
%   instead. Multi-dimensional counterpart: previewPredictiveFeedbackMD.
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty, mustBeInteger, mustBeFinite, mustBeNonnegative} = []
    end
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    prior = obj.currentTransitionPrior();

    if isempty(q)
        W = prior;
    else
        qCol = min(q, size(obj.D.local_cue_matrix, 2));
        pcue = squeeze(obj.D.local_cue_matrix(:, qCol, :));
        if P == 1
            pcue = pcue(:);
        end
        W = obj.normalizeColumns(prior .* pcue);
    end

    Mstate = obj.D.retention .* obj.D.state_filtered_mean + obj.D.drift;
    Vstate = obj.D.retention.^2 .* obj.D.state_filtered_var + obj.sigma_process_noise^2;
    for p = 1:P
        novel = min(obj.D.C(p) + 1, Cmax);
        if obj.D.C(p) < obj.max_contexts
            Mstate(novel,p) = obj.stationaryStateMean(obj.D.retention(novel,p), obj.D.drift(novel,p));
            Vstate(novel,p) = obj.stationaryStateVar(obj.D.retention(novel,p));
        end
    end
    M = Mstate + obj.D.bias;
    V = Vstate + obj.observationVariance();
end