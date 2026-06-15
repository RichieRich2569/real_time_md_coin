function [pmf, labels] = previewCuePmf(obj)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    Qn = size(obj.D.local_cue_matrix, 2);
    labels = 1:Qn;
    pmf = zeros(1, Qn);
    for p = 1:P
        prior = obj.D.local_transition_matrix(obj.D.context(p), :, p)';
        prior = obj.normalizeColumns(prior);
        cueGivenContext = obj.D.local_cue_matrix(:,:,p);
        if size(cueGivenContext,1) < Cmax
            cueGivenContext(Cmax,Qn) = 0;
        end
        pmf = pmf + (prior' * cueGivenContext);
    end
    pmf = pmf ./ P;
    if sum(pmf) > 0
        pmf = pmf ./ sum(pmf);
    end
end
