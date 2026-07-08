function [assignment, prototypes, usedSeed] = initializeContextAlignment(obj, Km, modalIdx)
%INITIALIZECONTEXTALIGNMENT Seed the assignment and prototypes for alignment.
%   [assignment, prototypes, usedSeed] = initializeContextAlignment(obj, Km,
%   modalIdx) produces the starting point for optimizeContextAlignment.m.
%
%   Two paths:
%     * Warm start (usedSeed = true): if the previous alignment stored in
%       obj.alignment_seed is compatible with the current cardinality Km (same K
%       and matching prototype shape - see isCompatibleSeed), reuse its
%       prototypes and per-particle assignment. This keeps global labels stable
%       across trials, which the warm-start test in test_global_alignment.m
%       relies on. Any modal particle left unlabelled by the seed is filled with
%       the identity map, and the novel-context slot (Km+1) is labelled when
%       there is room below max_contexts.
%     * Cold start (usedSeed = false): anchor on the first modal particle with
%       the identity map, then build prototypes from that single particle.
%
%   assignment is (max_contexts+1)-by-num_particles; column p maps local context
%   rows to global labels for particle p (0 = unassigned).

    Cmax = obj.max_contexts + 1;
    assignment = zeros(Cmax, obj.num_particles);
    usedSeed = false;

    if isCompatibleSeed(obj, Km)
        % --- Warm start from the previous alignment seed ---
        prototypes = obj.alignment_seed.global_contexts;
        if isfield(obj.alignment_seed, 'assignment') && ...
                isequal(size(obj.alignment_seed.assignment), size(assignment))
            assignment(:, modalIdx) = obj.alignment_seed.assignment(:, modalIdx);
        end
        for idx = 1:numel(modalIdx)
            p = modalIdx(idx);
            % Fill particles the seed did not cover with the identity map.
            if all(assignment(1:Km, p) == 0)
                assignment(1:Km, p) = 1:Km;
            end
            % Label the novel-context slot when there is room for it.
            if Km < obj.max_contexts && assignment(Km+1, p) == 0
                assignment(Km+1, p) = Km + 1;
            end
        end
        usedSeed = true;
        return;
    end

    % --- Cold start: anchor identity map on the first modal particle ---
    anchor = modalIdx(1);
    assignment(1:Km, anchor) = 1:Km;
    if Km < obj.max_contexts
        assignment(Km+1, anchor) = Km + 1;
    end
    prototypes = obj.updateGlobalContexts(Km, anchor, 1, assignment);
end

function tf = isCompatibleSeed(obj, Km)
%ISCOMPATIBLESEED True when the stored alignment seed can be reused for Km.
%   The seed is reusable only if it exists, targeted the same cardinality Km, and
%   carries prototypes whose fields/shape match the current model (scalar vs MD).

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
