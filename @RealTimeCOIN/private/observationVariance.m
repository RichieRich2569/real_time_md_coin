function v = observationVariance(obj)
%OBSERVATIONVARIANCE Scalar observation (measurement) noise variance.
%
%   v = observationVariance(obj) returns the total scalar observation noise
%   variance, the sum of the independent sensory and motor noise variances,
%       v = sigma_sensory_noise^2 + sigma_motor_noise^2.
%   This is the scalar quantity that observationNoiseCov.m scales into the
%   isotropic R = v * I_N for the multi-dimensional model.

    v = obj.sigma_sensory_noise^2 + obj.sigma_motor_noise^2;
end
