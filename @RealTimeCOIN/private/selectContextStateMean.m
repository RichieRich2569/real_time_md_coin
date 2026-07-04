function s = selectContextStateMean(obj, idx)
%SELECTCONTEXTSTATEMEAN Mean over particles of the state mean at a chosen context.
%
%   s = selectContextStateMean(obj, idx) takes a 1-by-P row of per-particle
%   context indices (e.g. the argmax of the responsibilities or predicted
%   probabilities) and returns the particle-average of obj.D.state_mean at
%   those contexts. Scalar for state_dim == 1; an N-by-1 vector otherwise.
%   Shared by explicit_component and the state_cstar* query methods so the
%   scalar/MD selection logic lives in one place.
    P = obj.num_particles;
    idx = idx(:)';

    if obj.state_dim == 1
        lin = sub2ind(size(obj.D.state_mean), idx, 1:P);
        s = mean(obj.D.state_mean(lin));
        return;
    end

    N = obj.state_dim;
    s = zeros(N, 1);
    for p = 1:P
        s = s + obj.D.state_mean(:, idx(p), p);
    end
    s = s ./ P;
end
