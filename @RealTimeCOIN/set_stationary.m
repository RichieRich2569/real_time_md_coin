function set_stationary(obj)
    resetParticles(obj);
    obj.pending_q = [];
    obj.trial = 0;
    obj.cue_values = [];
end
