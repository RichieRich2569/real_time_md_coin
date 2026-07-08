function instantiateCueIfNeeded(obj, q)
%INSTANTIATECUEIFNEEDED Split off a new global cue category via stick-breaking.
%   instantiateCueIfNeeded(obj, q) grows the global cue distribution to cover a
%   newly observed cue label q when q exceeds the current count obj.D.Q. It
%   applies one Beta(1, gamma_cue) stick-breaking step per particle: the mass
%   currently on category q is split into a retained fraction b (stays on q)
%   and (1 - b) (moves to the fresh category q+1), matching the DP-based cue
%   prior in the COIN model. No-op when q is empty or already instantiated.
    if isempty(q) || q <= obj.D.Q
        return;
    end
    obj.ensureCueColumn(q + 1);
    b = obj.betaSample(ones(1, obj.num_particles), obj.gamma_cue * ones(1, obj.num_particles));
    mass = obj.D.global_cue_probabilities(q, :);
    obj.D.global_cue_probabilities(q+1, :) = mass .* (1 - b);
    obj.D.global_cue_probabilities(q, :) = mass .* b;
    obj.D.Q = q;
end
