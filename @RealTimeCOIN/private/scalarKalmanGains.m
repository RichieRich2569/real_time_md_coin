function gains = scalarKalmanGains(obj)
%SCALARKALMANGAINS Per-particle scalar Kalman gains for every context.
%
%   gains = scalarKalmanGains(obj) returns the (max_contexts+1)-by-P matrix of
%   scalar Kalman gains, state_var ./ state_feedback_var, as used in the COIN
%   filter update. This is a pure read of obj.D and mirrors the gain term the
%   inference pipeline forms internally; it does not mutate the model state.
%
%   Shared by the scalar-model c* Kalman-gain query methods (kalman_gain_cstar1
%   and kalman_gain_cstar2) so the gain formula lives in one place. Scalar model
%   only (state_dim == 1); callers guard with mustBeScalarModel.
    gains = obj.D.state_var ./ obj.D.state_feedback_var;
end
