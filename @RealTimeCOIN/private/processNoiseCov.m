function Q = processNoiseCov(obj)
%PROCESSNOISECOV State (process) noise covariance Q for the MD model.
%
%   Returns the explicitly supplied process_noise_covariance when set,
%   otherwise the isotropic default sigma_process_noise^2 * I_N. This is the
%   multi-dimensional generalisation of the scalar process variance used in
%   predictStates.m (qv = sigma_process_noise^2). At N == 1 the default
%   collapses to that scalar, preserving the original behaviour.

    N = obj.state_dim;
    if isempty(obj.process_noise_covariance)
        Q = obj.sigma_process_noise^2 * eye(N);
    else
        Q = (obj.process_noise_covariance + obj.process_noise_covariance') ./ 2;
    end
end
