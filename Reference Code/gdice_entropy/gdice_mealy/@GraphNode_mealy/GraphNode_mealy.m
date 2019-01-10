classdef GraphNode_mealy < handle
    properties
        idxNode;
        myTMA; %TMA to be executed in this node
                  % 1   2   3   4   5   6   7  * 3  health states = 21 obs
                  %000 110 120 130 210 220 230 
        nextNode; %1-by-numObs vector for outgoing edges, indicating which node idx to connect to next
%         pVectorTMA; %MOORE REPRESENTATION ONLY! DO NOT USE! numTMAs-by-1 vector, representing probability of choosing a TMA
        pTableTMA; %numTMAs-by-numObs matrix, representing probability of choosing a TMA
        pTableNextNode; %numNodesInFullGraph-by-numObs matrix, representing probability of choosing the next graph node based on observation received
        numTMAs;
        numObs;
        numNodesInFullGraph;
        
        TMAs; %vector of sampled TMAS
        transitions; %matrix of sampled next-node transitions
    end
    
    methods
        function obj = GraphNode_mealy(numNodesInFullGraph, idxNode, numTMAs, numObs, numSamples)
            obj.idxNode = idxNode;
            obj.numTMAs = numTMAs;
            obj.numObs = numObs;
            obj.numNodesInFullGraph = numNodesInFullGraph;
            obj.transitions = zeros(numSamples, numObs);
            obj.TMAs = zeros(numSamples, numObs);
            
%             obj.pVectorTMA = ones(numTMAs, 1)/numTMAs;
            obj.pTableTMA = ones(numTMAs, numObs)/numTMAs;
            obj.pTableNextNode = ones(numNodesInFullGraph, numObs)/numNodesInFullGraph;
        end
        
        function sampleTMAs(obj, idxObs, numSamples)
%             obj.TMAs = discretesample(obj.pVectorTMA, numSamples)';
            obj.TMAs(:,idxObs) = discretesample(obj.pTableTMA(:,idxObs), numSamples);
        end
        
        function sampleTrans(obj, idxObs, numSamples)
            obj.transitions(:,idxObs) = discretesample(obj.pTableNextNode(:,idxObs), numSamples);
        end
        
        function setToSampleNumber(obj, idxSample)
            obj.myTMA = obj.TMAs(idxSample,:);
            obj.nextNode = obj.transitions(idxSample,:);
        end
        
%         function setTMA(obj, TMA)
%             obj.myTMA = TMA;
%         end
        
        function setNextNode(obj, obsIdx, nextNodeIdx)
            %obsIdx corresponds to those defined in properties above
            obj.nextNode(obsIdx) = nextNodeIdx;
        end
    end
end