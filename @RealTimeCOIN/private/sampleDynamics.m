function sampleDynamics(obj)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    priorPrec = diag([obj.prior_precision_retention, obj.prior_precision_drift]);
    priorMean = [obj.prior_mean_retention; obj.prior_mean_drift];
    qVar = obj.sigma_process_noise^2;
    for p = 1:P
        for c = 1:Cmax
            ss2 = squeeze(obj.D.dynamics_ss_2(c,p,:,:));
            ss1 = squeeze(obj.D.dynamics_ss_1(c,p,:));
            if qVar == 0
                covar = obj.safeInverse(priorPrec + ss2 ./ eps);
                mu = covar * (priorPrec * priorMean + ss1 ./ eps);
            else
                covar = obj.safeInverse(priorPrec + ss2 ./ qVar);
                mu = covar * (priorPrec * priorMean + ss1 ./ qVar);
            end
            sample = obj.sampleBivariateTruncated(mu, covar);
            obj.D.retention(c,p) = sample(1);
            obj.D.drift(c,p) = sample(2);
            obj.D.dynamics_mean(:,c,p) = mu;
            obj.D.dynamics_covar(:,:,c,p) = covar;
        end
    end
end
