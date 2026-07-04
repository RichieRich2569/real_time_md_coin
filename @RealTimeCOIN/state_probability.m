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
        values = values(:)';
        densities = singleStateDensity(obj, values, ...
            obj.D.responsibilities, obj.D.state_filtered_mean, obj.D.state_filtered_var);
        return;
    end

    densities = multiStateDensity(obj, values, ...
        obj.D.responsibilities, obj.D.state_filtered_mean, obj.D.state_filtered_cov);
end

function densities = singleStateDensity(obj, values, W, M, V)
%SINGLESTATEDENSITY Gaussian-mixture density at K query points (scalar model).
        arguments
            obj (1, 1) RealTimeCOIN
            values (1, :) double {mustBeFinite, mustBeReal}
            W (:, :) double {mustBeFinite, mustBeNonnegative, mustMarginalize(obj, W)}
            M (:, :) double {mustBeFinite, mustBeReal}
            V (:, :) double {mustBeFinite, mustBeNonnegative}
        end
        densities = zeros(size(values));
        for p = 1:obj.num_particles
            for c = 1:(obj.max_contexts+1)
                if W(c,p) > 0
                    densities = densities + W(c,p) .* obj.normal_pdf(values, M(c,p), V(c,p));
                end
            end
        end
        densities = densities ./ obj.num_particles;
end

function densities = multiStateDensity(obj, values, W, M, V)
%MULTISTATEDENSITY Gaussian-mixture density at N-by-K query points (MD model).
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
        W (:, :) double {mustBeFinite, mustBeNonnegative, mustMarginalize(obj, W)}
        M (:, :, :) double {mustBeFinite, mustBeReal}
        V (:, :, :, :) double {mustBeFinite, mustBeCovarianceMatrix}
    end
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

function mustMarginalize = mustMarginalize(obj, prob)
    % Checks if a vector of [C,P] probabilities are marginalized over C.
    arguments
        obj (1, 1) RealTimeCOIN
        prob (:, :) double {mustBeFinite, mustBeNonnegative}
    end
    mustMarginalize = true;
    if size(prob, 1) == obj.max_contexts + 1
        if any(abs(sum(prob, 1) - 1) > 1e-6)
            mustMarginalize = false;
        end
    end
end

function mustBeCovarianceMatrix(V)
    % Fast page-wise covariance screen for V(:,:,c,p).
    if ~isreal(V)
        error('RealTimeCOIN:InvalidCovarianceMatrix', ...
            'Covariance matrices must be real.');
    end

    N = size(V, 1);
    if size(V, 2) ~= N
        error('RealTimeCOIN:InvalidCovarianceMatrix', ...
            'Each covariance matrix must be square; received %d-by-%d pages.', ...
            size(V, 1), size(V, 2));
    end

    if isempty(V)
        return;
    end

    tol = 1e-10 * max(1, max(abs(V), [], 'all'));
    if any(abs(V - permute(V, [2 1 3 4])) > tol, 'all')
        error('RealTimeCOIN:InvalidCovarianceMatrix', ...
            'Each covariance matrix must be symmetric.');
    end

    pages = reshape(V, N * N, []);
    variances = pages(1:(N + 1):end, :);
    if any(variances < -tol, 'all')
        error('RealTimeCOIN:InvalidCovarianceMatrix', ...
            'Covariance matrix variances must be nonnegative.');
    end

    if N > 1
        sigma = sqrt(max(variances, 0));
        covLimit = reshape(sigma, N, 1, []) .* reshape(sigma, 1, N, []);
        if any(abs(reshape(V, N, N, [])) - covLimit > tol, 'all')
            error('RealTimeCOIN:InvalidCovarianceMatrix', ...
                'Covariances must satisfy |cov(i,j)| <= sqrt(var(i)*var(j)).');
        end
    end
end
