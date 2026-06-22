function validateMultiDimConfig(obj)
%VALIDATEMULTIDIMCONFIG Validate the multi-dimensional configuration.
%
%   Checks any user-supplied process/observation noise covariances against
%   state_dim and the symmetric positive-semidefinite requirement. Called
%   once from the constructor so that misconfigurations fail fast with clear
%   identifiers rather than surfacing as obscure linear-algebra errors deep
%   in the particle filter. When state_dim == 1 the scalar pipeline is used
%   and the covariance overrides are ignored (a warning is issued if set).

    N = obj.state_dim;

    if N == 1
        if ~isempty(obj.process_noise_covariance) || ~isempty(obj.observation_noise_covariance)
            warning('RealTimeCOIN:CovarianceIgnored', ...
                ['process_noise_covariance/observation_noise_covariance are ', ...
                 'ignored when state_dim == 1; the scalar sigma_* properties ', ...
                 'are used instead.']);
        end
        return;
    end

    validateNoiseCovariance(obj.process_noise_covariance, N, 'process_noise_covariance');
    validateNoiseCovariance(obj.observation_noise_covariance, N, 'observation_noise_covariance');
end

function validateNoiseCovariance(M, N, name)
%VALIDATENOISECOVARIANCE Assert that M is empty or a valid N-by-N SPD-ish matrix.
    if isempty(M)
        return; % Empty selects the isotropic default elsewhere.
    end
    if ~isequal(size(M), [N, N])
        error('RealTimeCOIN:BadCovarianceSize', ...
            '%s must be a %d-by-%d matrix to match state_dim, but is %d-by-%d.', ...
            name, N, N, size(M,1), size(M,2));
    end
    if any(~isfinite(M(:)))
        error('RealTimeCOIN:BadCovarianceValue', ...
            '%s must contain only finite values.', name);
    end
    asym = max(abs(M - M'), [], 'all');
    if asym > 1e-9 * max(1, max(abs(M), [], 'all'))
        error('RealTimeCOIN:CovarianceNotSymmetric', ...
            '%s must be symmetric (max asymmetry %.3g).', name, asym);
    end
    % Positive-semidefinite check on the symmetrized matrix; a small negative
    % tolerance absorbs round-off in user-supplied matrices.
    Msym = (M + M') ./ 2;
    minEig = min(eig(Msym));
    if minEig < -1e-9 * max(1, max(abs(M), [], 'all'))
        error('RealTimeCOIN:CovarianceNotPSD', ...
            '%s must be positive semidefinite (min eigenvalue %.3g).', name, minEig);
    end
end
