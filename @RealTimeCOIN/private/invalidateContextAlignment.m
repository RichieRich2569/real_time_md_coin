function invalidateContextAlignment(obj)
    obj.state_version = obj.state_version + 1;
    obj.alignment_cache = [];
end
