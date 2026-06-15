function saveModel(obj, filename, setStationary)
    if nargin < 3
        setStationary = true;
    end
    if setStationary
        saved = serializableState(obj);
        obj.set_stationary();
        model = serializableState(obj);
        save(filename, 'model');
        restoreSerializableState(obj, saved);
    else
        model = serializableState(obj);
        save(filename, 'model');
    end
end
