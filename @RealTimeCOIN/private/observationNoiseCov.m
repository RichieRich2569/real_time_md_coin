function R = observationNoiseCov(obj)
%OBSERVATIONNOISECOV Observation (measurement) noise covariance R for the MD model.
%
%   Returns the explicitly supplied observation_noise_covariance when set,
%   otherwise the isotropic default observationVariance() * I_N, where
%   observationVariance() = sigma_sensory_noise^2 + sigma_motor_noise^2. This
%   is the multi-dimensional generalisation of the scalar observation variance
%   and collapses to it at N == 1.

    N = obj.state_dim;
    if isempty(obj.observation_noise_covariance)
        R = obj.observationVariance() * eye(N);
    else
        R = (obj.observation_noise_covariance + obj.observation_noise_covariance') ./ 2;
    end
end
