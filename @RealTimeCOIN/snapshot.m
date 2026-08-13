function s = snapshot(obj)
%SNAPSHOT Capture the full model state as a plain, serialisable struct.
%   s = snapshot(obj) returns a plain struct holding everything needed to
%   reconstruct the model (every public property plus the private particle
%   store and bookkeeping). It is the public, in-memory counterpart of
%   saveModel/loadModel: use it to checkpoint a model and later restore it with
%   loadSnapshot, or to hand model state to/from parallel workers without disk
%   I/O.
%
%   The returned struct is a value (no shared handles), so a snapshot is a true
%   deep copy of the model state and is safe to serialise across parfor
%   boundaries. Round-trip guarantee: after loadSnapshot(other, snapshot(obj)),
%   other produces identical outputs from every query method as obj.
%
%   See also LOADSNAPSHOT, SAVEMODEL, LOADMODEL.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    s = serializableState(obj);
end
