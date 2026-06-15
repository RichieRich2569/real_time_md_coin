function q = localCueProbability(obj, local, p)
    q = squeeze(obj.D.local_cue_matrix(local,:,p));
    q = obj.normalizeProbability(q(:)');
end
