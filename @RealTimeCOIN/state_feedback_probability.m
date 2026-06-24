function densities = state_feedback_probability(obj, values)
%STATE_FEEDBACK_PROBABILITY Predictive feedback (observation) density on a grid.
%
%   For the scalar model (state_dim == 1) values is a vector and densities a
%   row vector of equal length; for the multi-dimensional model values is an
%   N-by-K matrix of column query points and densities is a 1-by-K row. The
%   density is the predictive Gaussian mixture over particles and contexts,
%       p(x) = (1/P) sum_p sum_c W(c,p) N(x | m_{c,p}, V_{c,p}),
%   using the predicted context probabilities and the predictive feedback
%   moments (state moments inflated by the observation noise).

    if obj.state_dim == 1
        values = values(:)';
        densities = zeros(size(values));
        W = obj.D.predicted_probabilities;
        M = obj.D.state_feedback_mean;
        V = obj.D.state_feedback_var;
        for p = 1:obj.num_particles
            for c = 1:(obj.max_contexts+1)
                if W(c,p) > 0
                    densities = densities + W(c,p) .* obj.normal_pdf(values, M(c,p), V(c,p));
                end
            end
        end
        densities = densities ./ obj.num_particles;
        return;
    end

    densities = multiFeedbackDensity(obj, values, ...
        obj.D.predicted_probabilities, obj.D.state_feedback_mean, obj.D.state_feedback_cov);
end

function densities = multiFeedbackDensity(obj, values, W, M, V)
%MULTIFEEDBACKDENSITY Gaussian-mixture density at N-by-K query points (MD model).
    N = obj.state_dim;
    if size(values, 1) ~= N
        error('RealTimeCOIN:GridDimensionMismatch', ...
            ['state_feedback_probability expects an %d-by-K grid (each column ', ...
             'a query point) for state_dim == %d; received a %d-by-%d array.'], ...
            N, N, size(values, 1), size(values, 2));
    end
    K = size(values, 2);
    densities = zeros(1, K);
    Cmax = obj.max_contexts + 1;
    for p = 1:obj.num_particles
        for c = 1:Cmax
            if W(c,p) > 0
                densities = densities + W(c,p) .* ...
                    obj.gaussianPdfColumnsMD(values, M(:,c,p), V(:,:,c,p));
            end
        end
    end
    densities = densities ./ obj.num_particles;
end
