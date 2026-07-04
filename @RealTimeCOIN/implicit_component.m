function i = implicit_component(obj)
%IMPLICIT_COMPONENT Implicit component of adaptation.
%
%   i = implicit_component(obj) returns the motor output minus the average
%   predicted state, i.e. the part of the motor output attributable to the
%   (across-context marginal) bias — COIN's implicit component of learning
%   (plot_implicit_component / plot_average_bias). Scalar for state_dim == 1;
%   an N-by-1 vector for the multi-dimensional model. Reflects obj.trial.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    [mu, ~] = obj.state_moments();
    i = obj.motor_output() - mu;
end
