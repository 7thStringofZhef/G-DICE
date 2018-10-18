classdef BeliefNode < handle
    properties
       idx; %unique idx for the node
       name; %string name
       xE; %environmental observation
    end
    methods
        function obj = BeliefNode(idx, name, xE)
            obj.idx = idx;
            obj.name = name;
            obj.xE = xE;
        end
        function xeSamplePackage(obj)
           obj.xE.samplePackage(); 
        end
        function xeSetPhi(obj, idxAgent, hasArrived, hasLeft)
            obj.xE.setPhi(idxAgent, hasArrived, hasLeft);
        end
    end
end