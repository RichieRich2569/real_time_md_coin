function densities = mixtureDensityOnGrid(obj, values, W, M, V, normalizer, methodName)
%MIXTUREDENSITYONGRID Weighted Gaussian-mixture density on a query grid.
%
%   densities = mixtureDensityOnGrid(obj, values, W, M, V, normalizer, ...
%   methodName) evaluates the Gaussian mixture
%       p(x) = (1/normalizer) sum_p sum_c W(c,p) N(x | M(.,c,p), V(.,c,p))
%   on a grid of query points, dispatching on obj.state_dim. It is the shared
%   scalar/MD mixture-on-grid loop behind state_probability,
%   state_feedback_probability, novel_state_probability and
%   novel_state_feedback_probability.
%
%   For the scalar model (state_dim == 1) values is reshaped to a row vector, M
%   and V are (C-by-P) means/variances and each component uses obj.normal_pdf.
%   For the multi-dimensional model values is the N-by-K grid (columns are query
%   points, validated against state_dim), M is N-by-C-by-P, V is N-by-N-by-C-by-P
%   and each component uses obj.gaussianPdfColumnsMD. The double loop runs p
%   (outer) over size(W, 2) then c (inner) over size(W, 1), skipping zero-weight
%   components, so the accumulation order is byte-identical to the per-method
%   loops it replaces. Division by normalizer is applied only when normalizer > 0
%   so an empty mixture returns the zero density unchanged.
%
%   methodName labels the caller in the grid-dimension-mismatch error. This is a
%   read-only evaluation: it uses no randomness and mutates no model state.
    if obj.state_dim == 1
        values = values(:)';
        densities = zeros(size(values));
        for p = 1:size(W, 2)
            for c = 1:size(W, 1)
                if W(c, p) > 0
                    densities = densities + W(c, p) .* obj.normal_pdf(values, M(c, p), V(c, p));
                end
            end
        end
        if normalizer > 0
            densities = densities ./ normalizer;
        end
        return;
    end

    N = obj.state_dim;
    if size(values, 1) ~= N
        error("RealTimeCOIN:GridDimensionMismatch", ...
            "%s expects an %d-by-K grid (each column a query point) for " + ...
            "state_dim == %d; received a %d-by-%d array.", ...
            methodName, N, N, size(values, 1), size(values, 2));
    end
    K = size(values, 2);
    densities = zeros(1, K);
    for p = 1:size(W, 2)
        for c = 1:size(W, 1)
            if W(c, p) > 0
                densities = densities + W(c, p) .* ...
                    obj.gaussianPdfColumnsMD(values, M(:, c, p), V(:, :, c, p));
            end
        end
    end
    if normalizer > 0
        densities = densities ./ normalizer;
    end
end
