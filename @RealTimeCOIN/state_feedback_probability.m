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
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end

    if obj.state_dim == 1
        densities = mixtureDensityOnGrid(obj, values, obj.D.predicted_probabilities, ...
            obj.D.state_feedback_mean, obj.D.state_feedback_var, ...
            obj.num_particles, "state_feedback_probability");
        return;
    end

    densities = mixtureDensityOnGrid(obj, values, obj.D.predicted_probabilities, ...
        obj.D.state_feedback_mean, obj.D.state_feedback_cov, ...
        obj.num_particles, "state_feedback_probability");
end
