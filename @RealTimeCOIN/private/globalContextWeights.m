function Wg = globalContextWeights(obj, W, alignment)
    if nargin < 3
        alignment = obj.ensureContextAlignment();
    end
    Cmax = obj.max_contexts + 1;
    Km = alignment.K;
    modalIdx = alignment.modal_particle_indices;
    Wg = zeros(Cmax, numel(modalIdx));
    for idx = 1:numel(modalIdx)
        p = modalIdx(idx);
        for local = 1:Cmax
            target = alignment.assignment(local,p);
            if target > 0 && target <= Cmax
                Wg(target,idx) = Wg(target,idx) + W(local,p);
            end
        end
        if Km >= obj.max_contexts
            Wg(Km+1:end,idx) = 0;
        end
    end
end
