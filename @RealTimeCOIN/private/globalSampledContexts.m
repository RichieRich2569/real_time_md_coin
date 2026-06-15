function c = globalSampledContexts(obj, alignment)
    if nargin < 2
        alignment = obj.ensureContextAlignment();
    end
    modalIdx = alignment.modal_particle_indices;
    c = NaN(1, numel(modalIdx));
    for idx = 1:numel(modalIdx)
        p = modalIdx(idx);
        local = obj.D.context(p);
        if local <= size(alignment.assignment, 1)
            target = alignment.assignment(local,p);
            if target > 0
                c(idx) = target;
            end
        end
    end
end
