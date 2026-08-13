function densities = state_probability(obj, values)
%STATE_PROBABILITY Posterior latent-state density on a grid of query points.
%
%   For the scalar model (state_dim == 1) values is a vector of points and the
%   returned densities is a row vector of the same length. For the
%   multi-dimensional model values is an N-by-K matrix whose columns are query
%   points and densities is a 1-by-K row. In both cases the density is the
%   posterior Gaussian mixture over particles and contexts,
%       p(x) = (1/P) sum_p sum_c W(c,p) N(x | m_{c,p}, V_{c,p}),
%   using the responsibilities and filtered (posterior) state moments.
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end

    if obj.state_dim == 1
        densities = mixtureDensityOnGrid(obj, values, obj.D.responsibilities, ...
            obj.D.state_filtered_mean, obj.D.state_filtered_var, ...
            obj.num_particles, "state_probability");
        return;
    end

    densities = mixtureDensityOnGrid(obj, values, obj.D.responsibilities, ...
        obj.D.state_filtered_mean, obj.D.state_filtered_cov, ...
        obj.num_particles, "state_probability");
end
