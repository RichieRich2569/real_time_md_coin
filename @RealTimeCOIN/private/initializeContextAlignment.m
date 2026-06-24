function [assignment, prototypes, usedSeed] = initializeContextAlignment(obj, Km, modalIdx)
    Cmax = obj.max_contexts + 1;
    assignment = zeros(Cmax, obj.num_particles);
    usedSeed = false;

    if isCompatibleSeed(obj, Km)
        prototypes = obj.alignment_seed.global_contexts;
        if isfield(obj.alignment_seed, 'assignment') && ...
                isequal(size(obj.alignment_seed.assignment), size(assignment))
            assignment(:, modalIdx) = obj.alignment_seed.assignment(:, modalIdx);
        end
        for idx = 1:numel(modalIdx)
            p = modalIdx(idx);
            if all(assignment(1:Km, p) == 0)
                assignment(1:Km, p) = 1:Km;
            end
            if Km < obj.max_contexts && assignment(Km+1, p) == 0
                assignment(Km+1, p) = Km + 1;
            end
        end
        usedSeed = true;
        return;
    end

    anchor = modalIdx(1);
    assignment(1:Km, anchor) = 1:Km;
    if Km < obj.max_contexts
        assignment(Km+1, anchor) = Km + 1;
    end
    prototypes = obj.updateGlobalContexts(Km, anchor, 1, assignment);
end

function tf = isCompatibleSeed(obj, Km)
    tf = ~isempty(obj.alignment_seed) && ...
        isfield(obj.alignment_seed, 'K') && obj.alignment_seed.K == Km && ...
        isfield(obj.alignment_seed, 'global_contexts');
    if ~tf
        return;
    end
    proto = obj.alignment_seed.global_contexts;
    if obj.state_dim > 1
        tf = isfield(proto, 'state_mean') && isfield(proto, 'state_cov') && ...
            isfield(proto, 'theta_mean') && size(proto.state_mean, 2) == Km;
    else
        tf = isfield(proto, 'state_mean') && isfield(proto, 'state_var') && ...
            isfield(proto, 'dynamics_mean') && numel(proto.state_mean) == Km;
    end
end
