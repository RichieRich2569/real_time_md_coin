function densities = state_feedback_given_context_probability(obj, values)
%STATE_FEEDBACK_GIVEN_CONTEXT_PROBABILITY Per-context feedback density on a grid.
%
%   Feedback (observation-space) counterpart of state_given_context_probability.
%   Returns a containers.Map keyed by global context label, each value that
%   context's predictive feedback density y = state + bias + noise evaluated at
%   the query points. For the scalar model (state_dim == 1) values is a vector
%   and each density is a row vector; for the multi-dimensional model values is
%   an N-by-K matrix of column query points and each density is a 1-by-K row.
%
%   Densities use the aligned global-context prototype moments, with the mean
%   shifted by the learned observation bias and the covariance inflated by the
%   observation noise R (mirroring predictStateFeedback / predictStateFeedbackMD:
%   fbMean = state_mean + bias, fbCov = state_cov + R).
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end

    if size(values, 1) ~= obj.state_dim
        error('RealTimeCOIN:GridDimensionMismatch', ...
            ['state_feedback_given_context_probability expects an %d-by-K grid ', ...
             '(each column a query point) for state_dim == %d; received a %d-by-%d array.'], ...
            obj.state_dim, obj.state_dim, size(values, 1), size(values, 2));
    end

    densities = containers.Map('KeyType', 'double', 'ValueType', 'any');
    alignment = ensureContextAlignment(obj);
    active = activeSummaryContexts(obj);
    active = active(active <= alignment.K);
    if isempty(active) && alignment.K > 0
        active = 1;
    end

    if obj.state_dim == 1
        values = values(:)';
        M = alignment.global_contexts.state_mean;
        V = alignment.global_contexts.state_var;
        B = alignment.global_contexts.bias_mean;
        R = obj.sigma_sensory_noise^2;
        for c = active
            densities(c) = obj.normal_pdf(values, M(c) + B(c), V(c) + R);
        end
        return;
    end

    M = alignment.global_contexts.state_mean;     % N-by-Km
    V = alignment.global_contexts.state_cov;      % N-by-N-by-Km
    B = alignment.global_contexts.bias_mean;      % N-by-Km
    R = observationNoiseCov(obj);
    for c = active
        densities(c) = gaussianPdfColumnsMD(obj, values, M(:,c) + B(:,c), V(:,:,c) + R);
    end
end
