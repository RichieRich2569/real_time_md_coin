function s = makeMemberStream(generator, R, k, seed)
%MAKEMEMBERSTREAM Create the dedicated RNG substream for member k.
%   s = makeMemberStream(generator, R, k, seed) returns a RandStream that is
%   substream k of an R-way independent set derived from seed. Because the
%   stream is a pure function of (generator, R, k, seed) -- not of execution
%   order, worker placement, or the number of cores -- member k's randomness is
%   reproducible and independent of the other members. Used both when building
%   the live members and when replaying members in simulate().
    s = RandStream.create(generator, 'NumStreams', R, ...
        'StreamIndices', k, 'Seed', seed);
end
