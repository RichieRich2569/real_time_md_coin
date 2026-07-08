function k = kalman_gain_cstar1(obj)
%KALMAN_GAIN_CSTAR1 Kalman gain of the highest-responsibility context.
%
%   k = kalman_gain_cstar1(obj) selects, in each particle, the context with the
%   highest responsibility (c*1) and reads off its scalar Kalman gain
%   (state_var ./ state_feedback_var, as in the COIN filter update), then
%   averages over particles. Mirrors COIN's plot_Kalman_gain_given_cstar1.
%   Reflects the model state as of obj.trial.
%
%   Defined for the scalar model only (state_dim == 1); the Kalman gain is a
%   matrix in the multi-dimensional model and has no scalar counterpart.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    mustBeScalarModel(obj, 'kalman_gain_cstar1');
    P = obj.num_particles;
    gains = scalarKalmanGains(obj);
    [~, idx] = max(obj.D.responsibilities, [], 1);
    lin = sub2ind(size(gains), idx, 1:P);
    k = mean(gains(lin));
end
