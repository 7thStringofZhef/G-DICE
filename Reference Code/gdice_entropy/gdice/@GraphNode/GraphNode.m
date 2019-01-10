classdef GraphNode < handle
    properties
        idxNode;
        myTMA; %TMA to be executed in this node
        nextNode; %1-by-numObs vector for outgoing edges, indicating which node idx to connect to next
        pVectorTMA; %numTMAs-by-1 vector, representing probability of choosing a TMA
        pTableNextNode; %numNodesInFullGraph-by-numObs matrix, representing probability of choosing the next graph node based on observation received
        numTMAs;
        numObs;
        numNodesInFullGraph;
        
        
        TMAs; %vector of sampled TMAS
        transitions; %matrix of sampled next-node transitions
    end
    
    methods
        function obj = GraphNode(numNodesInFullGraph, idxNode, numTMAs, numObs, numSamples)
            obj.idxNode = idxNode;
            obj.numTMAs = numTMAs;
            obj.numObs = numObs;
            obj.numNodesInFullGraph = numNodesInFullGraph;
            obj.transitions = zeros(numSamples, numObs);
%             obj.obs_counter = zeros(n_bins_xy,n_bins_xy);
            
            obj.pVectorTMA = ones(numTMAs, 1)/numTMAs;
            obj.pTableNextNode = ones(numNodesInFullGraph, numObs)/numNodesInFullGraph;
        end
        
        function sampleTMAs(obj, numSamples)
            obj.TMAs = discretesample(obj.pVectorTMA, numSamples)';
        end
        
        function sampleTrans(obj, idxObs, numSamples)
            obj.transitions(:,idxObs) = discretesample(obj.pTableNextNode(:,idxObs), numSamples);
        end
        
        function setToSampleNumber(obj, idxSample)
            obj.myTMA = obj.TMAs(idxSample);
            obj.nextNode = obj.transitions(idxSample,:);
        end
        
        function setTMA(obj, TMA)
            obj.myTMA = TMA;
        end
        
        function setNextNode(obj, obsIdx, nextNodeIdx)
            %obsIdx corresponds to those defined in properties above
            obj.nextNode(obsIdx) = nextNodeIdx;
        end
    end
end