function assertScalarOnly(obj, name)
%ASSERTSCALARONLY Guard for query methods not yet generalised to MD.
%
%   Grid-based predictive densities and the univariate predictive CDF are
%   inherently scalar and have no agreed multivariate form yet. They raise a
%   clear, actionable error for state_dim > 1 rather than producing a
%   misleading scalar result. Multi-dimensional summaries are available via
%   state_moments, predictive_feedback_moments, motor_output and diagnostics.

    if obj.state_dim > 1
        error('RealTimeCOIN:MultiDimQueryUnsupported', ...
            ['%s currently supports only scalar state (state_dim == 1). For ', ...
             'multi-dimensional summaries use state_moments(), ', ...
             'predictive_feedback_moments(), motor_output() or diagnostics().'], name);
    end
end
