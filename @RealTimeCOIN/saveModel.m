function saveModel(obj, filename, setStationary)
%SAVEMODEL Serialise the model state to a .mat file.
%   saveModel(obj, filename) writes a serialisable snapshot of the model to
%   the file named filename (a "model" variable inside the .mat). By default
%   the model is first placed at its stationary prior via set_stationary, so
%   the saved snapshot is contingency-independent; the live object is restored
%   to its pre-save state afterwards.
%
%   saveModel(obj, filename, setStationary) controls that behaviour. When
%   setStationary is false the current (in-progress) state is saved verbatim
%   without calling set_stationary.
%
%   The temporary stationary reset is wrapped in try/catch so the live object
%   is restored even if serialisation fails. Reload with loadModel.
%
%   See also LOADMODEL, SET_STATIONARY.
    arguments
        obj (1, 1) RealTimeCOIN
        filename (1, :) char {mustBeNonempty}
        setStationary (1, 1) logical = true
    end
    if setStationary
        saved = serializableState(obj);
        try
            obj.set_stationary();
            model = serializableState(obj);
            save(filename, 'model');
            restoreSerializableState(obj, saved);
        catch err
            restoreSerializableState(obj, saved);
            rethrow(err);
        end
    else
        model = serializableState(obj);
        save(filename, 'model');
    end
end
