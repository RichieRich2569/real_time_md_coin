function Tg = globalTransitionTensor(obj, T, alignment)
    if nargin < 3
        alignment = obj.ensureContextAlignment();
    end
    Cmax = obj.max_contexts + 1;
    modalIdx = alignment.modal_particle_indices;
    Tg = zeros(Cmax, Cmax, numel(modalIdx));
    for idx = 1:numel(modalIdx)
        p = modalIdx(idx);
        for localFrom = 1:Cmax
            globalFrom = alignment.assignment(localFrom,p);
            if globalFrom <= 0 || globalFrom > Cmax
                continue;
            end
            for localTo = 1:Cmax
                globalTo = alignment.assignment(localTo,p);
                if globalTo > 0 && globalTo <= Cmax
                    Tg(globalFrom,globalTo,idx) = Tg(globalFrom,globalTo,idx) + T(localFrom,localTo,p);
                end
            end
        end
    end
end
