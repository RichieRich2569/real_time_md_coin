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

    if obj.state_dim == 1
        values = values(:)';
        densities = zeros(size(values));
        W = obj.D.responsibilities;
        M = obj.D.state_filtered_mean;
        V = obj.D.state_filtered_var;
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

    densities = multiStateDensity(obj, values, ...
        obj.D.responsibilities, obj.D.state_filtered_mean, obj.D.state_filtered_cov);
end

function densities = multiStateDensity(obj, values, W, M, V)
%MULTISTATEDENSITY Gaussian-mixture density at N-by-K query points (MD model).
    N = obj.state_dim;
    if size(values, 1) ~= N
        error('RealTimeCOIN:GridDimensionMismatch', ...
            ['state_probability expects an %d-by-K grid (each column a query ', ...
             'point) for state_dim == %d; received a %d-by-%d array.'], ...
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
