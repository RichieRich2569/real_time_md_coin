function assertScalarOnly(obj, name)
%ASSERTSCALARONLY Guard for query methods not yet generalised to MD.
%
%   Grid-based predictive densities and the univariate predictive CDF are
%   inherently scalar and have no agreed multivariate form yet. They raise a
%   clear, actionable error for state_dim > 1 rather than producing a
%   misleading scalar result. Multi-dimensional summaries are available via
%   state_moments, predictive_feedback_moments, motor_output and diagnostics.

    % Shares the RealTimeCOIN:ScalarModelOnly identifier with mustBeScalarModel
    % so callers can catch either scalar-only guard with a single identifier.
    if obj.state_dim > 1
        error('RealTimeCOIN:ScalarModelOnly', ...
            ['%s is only defined for the scalar model (state_dim == 1); ', ...
             'state_dim == %d. For multi-dimensional summaries use ', ...
             'state_moments(), predictive_feedback_moments(), motor_output() ', ...
             'or diagnostics().'], name, obj.state_dim);
    end
end
