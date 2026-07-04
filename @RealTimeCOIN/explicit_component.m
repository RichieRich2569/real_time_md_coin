function e = explicit_component(obj)
%EXPLICIT_COMPONENT Explicit component of adaptation.
%
%   e = explicit_component(obj) returns the predictive latent-state mean of the
%   highest-responsibility context, averaged over particles — COIN's explicit
%   component of learning (plot_explicit_component). On the first trial, when
%   responsibilities are not yet informative, context 1 is used, matching COIN.
%   Scalar for state_dim == 1; an N-by-1 vector for the multi-dimensional model.
%
%   This is the c*1 state (see state_cstar1); the implicit component
%   (implicit_component) is the residual of the motor output beyond this.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    if obj.trial <= 1
        idx = ones(1, obj.num_particles);
    else
        [~, idx] = max(obj.D.responsibilities, [], 1);
    end
    e = selectContextStateMean(obj, idx);
end
