classdef GraphPolicyController_mealy < handle
    properties
        nodes = [];
        numNodes;
        alpha; %learning rate
        numObs;
        numTMAs;
        numSamples;
        shouldUseAdaptiveLearning;
        shouldUpdateWorstEliteSample;
        shouldInjectNoise;
        shouldInjectNoiseUsingMaximalEntropy; 
        
        is_continuous_policy = false;
        obs_counter; % count the number of observations in each bin used for updating policy for discrete case, for analysis purposes only
    end
    
    methods
        function obj = GraphPolicyController_mealy(numNodes, alpha, numTMAs, numObs, numSamples, shouldUseAdaptiveLearning, shouldUpdateWorstEliteSample, shouldInjectNoise,shouldInjectNoiseUsingMaximalEntropy)
            obj.numNodes = numNodes;
            obj.alpha = alpha;
            obj.numObs = numObs;
            obj.numTMAs = numTMAs;
            obj.numSamples = numSamples;
            obj.shouldUseAdaptiveLearning = shouldUseAdaptiveLearning;
            obj.shouldUpdateWorstEliteSample = shouldUpdateWorstEliteSample;
            obj.shouldInjectNoise = shouldInjectNoise;
            obj.shouldInjectNoiseUsingMaximalEntropy = shouldInjectNoiseUsingMaximalEntropy;
            
            obj.obs_counter = zeros(numNodes,numObs);
            
            for idxNode = 1:obj.numNodes
                obj.appendToGraphNodes(idxNode, numTMAs, numObs, numSamples);
            end
        end
        
        function sample(obj, numSamples)
            for idxNode = 1:length(obj.nodes)
                for idxObs = 1:obj.numObs
                    %sample TMAs at the node, using updated pdf of best nodes
                    obj.nodes(idxNode).sampleTMAs(idxObs, numSamples);
                    
                    %sample node transitions, using updated pdf of best trasitions
                    obj.nodes(idxNode).sampleTrans(idxObs,numSamples);
                end
            end
        end
        
        function setGraph(obj, idxSample)
            for idxNode = 1:length(obj.nodes)
                %sets both TMA to be executed at node, and next-node transition
                obj.nodes(idxNode).setToSampleNumber(idxSample);
            end
        end
        
        function [worstEliteValue, maxValueIdxs, entropy_sums, just_injected_noise] = updateProbs(obj, curIterationValues, N_b, minValueAllowed, idxIteration, shouldInjectNoiseNow)
            output = true;
            just_injected_noise = false;
            
            %keep best N_b samples that have value better than
            %minValueAllowed
            [sortedValues, sortingIndices] = sort(curIterationValues,'descend');
            maxValues = sortedValues(1:N_b);
            maxValueIdxs = sortingIndices(1:N_b);
            maxValueIdxs = maxValueIdxs(maxValues>minValueAllowed);
            N_b = length(maxValueIdxs);
            if (output)
                fprintf('%d samples with value >%f found!\n', N_b,minValueAllowed);
            end
            
            %make sure you have non-zero number of best samples
            %this allows us to avoid re-normalization of pdfs, which we'd have to do if N_b = 0
            if (N_b > 0)
                weightPerSample = 1/N_b;
                 
                %go through each best sample of the N_b set
                for idxNode = 1:length(obj.nodes)
                    %this weighting takes care of the filtering/learning step automatically.                   
                    pTableTMA_new = obj.nodes(idxNode).pTableTMA.*(1-obj.alpha);
                    pTableNextNode_new = obj.nodes(idxNode).pTableNextNode.*(1-obj.alpha);%zeros(obj.numNodes, obj.numObs);
                    
                    for idxSample = maxValueIdxs'
                        if (output)
                            fprintf('Updating weights using "best" sample %d\n', idxSample);
                        end
                        
                        for idxObs = 1:obj.numObs
                            % update pTableTMA pdf at "best" TMA location
                            sampleTMA = obj.nodes(idxNode).TMAs(idxSample,idxObs);
                            pTableTMA_new(sampleTMA, idxObs) = pTableTMA_new(sampleTMA,idxObs) + weightPerSample*obj.alpha;
                        
                            % update pTableNextNode pdf at "best" next node locations
                            sampleNextNode = obj.nodes(idxNode).transitions(idxSample,idxObs);
                            pTableNextNode_new(sampleNextNode, idxObs) = pTableNextNode_new(sampleNextNode, idxObs) + weightPerSample*obj.alpha;
                        end
                    end
                    
                    %update the pdfs
                    obj.nodes(idxNode).pTableTMA = pTableTMA_new;
                    obj.nodes(idxNode).pTableNextNode = pTableNextNode_new;
                end
            end
            
            % add noise to transition pdfs to avoid degeneration
            if obj.shouldInjectNoise
                % calculate sum of per-transition-distribution entropies
                % for each node
                entropy_sums = zeros(length(obj.nodes),1);

                for idxNode = 1:length(obj.nodes)
%                     entColumnsPTableTMA = entropy_columnwise(obj.nodes(idxNode).pTableTMA, obj.numObs);
%                     entropy_sums(idxNode) = sum(entColumnsPTableTMA)/(entropy_maximal(obj.numNodes)*obj.numObs);

                    entColumnsPTableNextNode = entropy_columnwise(obj.nodes(idxNode).pTableNextNode,obj.numObs);
                    entropy_sums(idxNode) = sum(entColumnsPTableNextNode)/(entropy_maximal(obj.numNodes)*obj.numObs);
                end
                
                if shouldInjectNoiseNow
                    just_injected_noise = obj.injectNoise(idxIteration);
                end
            else
                entropy_sums = 0;
                just_injected_noise = false;
            end
            
            if (isempty(maxValueIdxs)||~obj.shouldUpdateWorstEliteSample)
                % do not change the minValueAllowed
                worstEliteValue = minValueAllowed;
            else
                % of the suitable elite samples (which exceed the previous
                % level set), choose the worst one as the next level set
                worstEliteValue = curIterationValues(maxValueIdxs(end));
            end
        end
        
        function just_injected_noise = injectNoise(obj,idxIteration)
            just_injected_noise = false;
            if obj.shouldInjectNoiseUsingMaximalEntropy
                % max possible entropy for categorical pdf of this size
                maxEnt = entropy_maximal(obj.numNodes);
                % degeneration noise injection rate (between 0 and 1)
                noise_injection_rate = 0.05;
                % entropy/maxEnt ratio at which to begin noise injection
                % % between 0 and 1. the higher this value, the earlier
                % injection begins.
                entFractionForInjection = 0.02; 
                
                for idxNode = 1:length(obj.nodes)
                    entColumnsPTableTMA = entropy_columnwise(obj.nodes(idxNode).pTableTMA,obj.numObs);
                    idxsDegenPTableTMA = entColumnsPTableTMA < maxEnt*entFractionForInjection;
                    obj.nodes(idxNode).pTableTMA(:,idxsDegenPTableTMA) = (1-noise_injection_rate)*obj.nodes(idxNode).pTableTMA(:,idxsDegenPTableTMA) + noise_injection_rate*ones(obj.numTMAs,sum(idxsDegenPTableTMA))./obj.numTMAs;
                    
                    entColumnsPTableNextNode = entropy_columnwise(obj.nodes(idxNode).pTableNextNode,obj.numObs);
                    idxsDegenPTableNextNode = entColumnsPTableNextNode < maxEnt*entFractionForInjection;
                    obj.nodes(idxNode).pTableNextNode(:,idxsDegenPTableNextNode) = (1-noise_injection_rate)*obj.nodes(idxNode).pTableNextNode(:,idxsDegenPTableNextNode) + noise_injection_rate*ones(obj.numNodes,sum(idxsDegenPTableNextNode))./obj.numNodes;
                    
                    if (sum(idxsDegenPTableTMA)>1 || sum(idxsDegenPTableNextNode)>1)
%                         fprintf(['idxNode: ' num2str(idxNode) '| sum(idxsDegenPTableTMA): ' num2str(sum(idxsDegenPTableTMA)) ' | sum(idxsDegenPTableNextNode): ' num2str(sum(idxsDegenPTableNextNode)) '\n'])
                        just_injected_noise = true;
                    end
                end
                if just_injected_noise
                    fprintf(['injected entropy\n'])
                end
            else
                for idxNode = 1:length(obj.nodes)
                    % add noise
                    noise_var = max(0.01-idxIteration/2000,0);
                    obj.nodes(idxNode).pTableTMA = obj.nodes(idxNode).pTableTMA + abs(normrnd(0,noise_var,obj.numTMAs,obj.numObs));
                    obj.nodes(idxNode).pTableNextNode = obj.nodes(idxNode).pTableNextNode + abs(normrnd(0,noise_var,obj.numNodes,obj.numObs));
                    
                    % normalize pdfs
                    column_wise_sums = sum(obj.nodes(idxNode).pTableTMA,1);
                    obj.nodes(idxNode).pTableTMA = obj.nodes(idxNode).pTableTMA./repmat(column_wise_sums,obj.numTMAs,1);
                    column_wise_sums = sum(obj.nodes(idxNode).pTableNextNode,1);
                    obj.nodes(idxNode).pTableNextNode = obj.nodes(idxNode).pTableNextNode./repmat(column_wise_sums,obj.numNodes,1);
                end
            end
        end
        
        function appendToGraphNodes(obj, idxNode, numTMAs, numObs, numSamples)
            newGraphNode = GraphNode_mealy(obj.numNodes, idxNode, numTMAs, numObs, numSamples);
            obj.nodes = [obj.nodes; newGraphNode];
        end
        
        function printNodes(obj)
            fprintf('GraphPolicyController_mealy nodes: %d, %d, %d, %d\n', obj.nodes(:).myTMA)
        end
        
        function [newPolicyNodeIdx, newTMAIdx] = getNextTMAIdx(obj, curNodeIdx, curXeIdx)
            %based on current node and received observation, move to next node in policy controller
            newPolicyNodeIdx = obj.nodes(curNodeIdx).nextNode(curXeIdx);
            %assign new TMA based on newly-assigned node in the policy controller
            newTMAIdx = obj.nodes(newPolicyNodeIdx).myTMA(curXeIdx);
            
            % record observation for analysis purposes only
            obj.obs_counter(curNodeIdx,curXeIdx) = obj.obs_counter(curNodeIdx,curXeIdx)+1;
        end
        
        %table getter
        function [TMAs, transitions] = getPolicyTable(obj)
            TMAs = zeros(obj.numNodes,1);
            transitions = zeros(obj.numNodes, obj.numObs);
            
            for idxNode = 1:obj.numNodes
                TMAs(idxNode) = obj.nodes(idxNode).myTMA;
                transitions(idxNode,:) = obj.nodes(idxNode).nextNode;
            end
        end
        
        %Pseudo-Mealy macrocontroller object getter
        function [TMAs, transitions] = getPseudoMealyPolicyTable(obj)
            TMAs = zeros(obj.numNodes,obj.numObs);
            transitions = zeros(obj.numNodes, obj.numObs);
            
            for idxNode = 1:obj.numNodes
                TMAs(idxNode,:) = obj.nodes(idxNode).myTMA;
                transitions(idxNode,:) = obj.nodes(idxNode).nextNode;
            end
            
%             TMAs = repmat(TMAs,1,obj.numObs);
        end
        
        %Pseudo-Mealy macrocontroller object getter
        function [TMAs, transitions] = getTmasAndTransProbs(obj)
            TMAs = zeros(obj.numNodes,obj.numObs);
            transitions = zeros(obj.numNodes, obj.numObs);
            
            for idxNode = 1:obj.numNodes
                TMAs(idxNode,:) = obj.nodes(idxNode).myTMA;
                transitions(idxNode,:) = obj.nodes(idxNode).nextNode;
            end
            
%             TMAs = repmat(TMAs,1,obj.numObs);
        end
        
        %table setter
        function setPolicyUsingTables(obj, TMAs, transitions)
            for idxNode = 1:obj.numNodes
                obj.nodes(idxNode).myTMA = TMAs(idxNode,:);
                obj.nodes(idxNode).nextNode = transitions(idxNode,:);
            end
        end
    end
end