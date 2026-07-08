function loadSnapshot(obj, s)
%LOADSNAPSHOT Restore full model state in place from a snapshot struct.
%   loadSnapshot(obj, s) rebuilds obj from a struct produced by snapshot,
%   overwriting every public property, the private particle store and the
%   trial/cue/alignment bookkeeping. It is the inverse of snapshot and the
%   in-memory counterpart of loadModel.
%
%   After loadSnapshot(obj, snapshot(other)), obj reproduces other's outputs
%   from every query method for the same inputs. The cached context alignment
%   is invalidated so it is recomputed on the next query.
%
%   See also SNAPSHOT, LOADMODEL, SAVEMODEL.
    arguments
        obj (1, 1) RealTimeCOIN
        s (1, 1) struct
    end
    restoreSerializableState(obj, s);
end
