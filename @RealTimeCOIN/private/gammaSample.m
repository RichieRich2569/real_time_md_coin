function g = gammaSample(~, shape)
%GAMMASAMPLE Draw gamma random variates with unit scale.
%   g = gammaSample(obj, shape) returns an array the same size as SHAPE whose
%   entries are independent draws from the gamma distribution with the given
%   shape parameters and unit scale, i.e. g(i) ~ Gamma(shape(i), 1). Entries
%   with a non-positive shape are returned as exactly 0 (the degenerate gamma),
%   so the result is always well defined for the counts and concentration
%   parameters passed in by the Beta/Dirichlet helpers that call this routine.
%   NaN shapes are treated the same as non-positive ones (NaN > 0 is false) and
%   map to 0.
%
%   The OBJ argument is unused: this is a stateless helper exposed as a class
%   method only so the gamma sampler can be swapped in a single place.
%
%   Inputs:
%     shape  Real numeric array of gamma shape parameters (any size).
%
%   Output:
%     g      Array the size of SHAPE of Gamma(shape, 1) draws; 0 where
%            shape <= 0 or shape is NaN.
%
%   This helper requires the built-in RANDG (available in base MATLAB and the
%   Statistics and Machine Learning Toolbox). A clear error is raised if RANDG
%   cannot be resolved so the failure is not mistaken for a modelling problem.
%
%   See also betaSample, dirichletSample, randg.

    if ~isnumeric(shape) || ~isreal(shape)
        error("RealTimeCOIN:gammaSample:invalidShape", ...
            "shape must be a real numeric array.");
    end

    g = zeros(size(shape));
    good = shape > 0;
    if any(good, 'all')
        % Guard against a missing RANDG only on the path that actually needs
        % it, so machines without the sampler still handle the all-degenerate
        % (shape <= 0) case unchanged.
        if exist('randg', 'builtin') ~= 5 && exist('randg', 'file') == 0
            error("RealTimeCOIN:gammaSample:randgUnavailable", ...
                "randg was not found on the MATLAB path. The gamma sampler " + ...
                "requires the built-in randg (base MATLAB or the Statistics " + ...
                "and Machine Learning Toolbox).");
        end
        g(good) = randg(shape(good));
    end
end
