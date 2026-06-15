function Xg = globalContextMatrix(obj, X, alignment)
    if nargin < 3
        alignment = obj.ensureContextAlignment();
    end
    Cmax = obj.max_contexts + 1;
    modalIdx = alignment.modal_particle_indices;
    Xg = zeros(Cmax, numel(modalIdx));
    for idx = 1:numel(modalIdx)
        p = modalIdx(idx);
        for local = 1:Cmax
            target = alignment.assignment(local,p);
            if target > 0 && target <= Cmax
                Xg(target,idx) = X(local,p);
            end
        end
    end
end
