function mustBeScalarModel(obj, methodName)
%MUSTBESCALARMODEL Error unless the model is scalar (state_dim == 1).
%
%   Guard for query methods whose quantity is intrinsically scalar-dynamics
%   specific (scalar Kalman gain, retention/drift-given-context densities) and
%   has no multi-dimensional counterpart, matching COIN.m which is scalar-only
%   for these.
    if obj.state_dim ~= 1
        error('RealTimeCOIN:ScalarModelOnly', ...
            '%s is only defined for the scalar model (state_dim == 1); state_dim == %d.', ...
            methodName, obj.state_dim);
    end
end
