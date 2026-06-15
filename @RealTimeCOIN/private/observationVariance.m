function v = observationVariance(obj)
    v = obj.sigma_sensory_noise^2 + obj.sigma_motor_noise^2;
end
