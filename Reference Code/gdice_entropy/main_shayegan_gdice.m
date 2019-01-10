%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Script to batch run simulations
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ************************************************************
clc
clear all
close all

addpath(genpath(pwd))
path_java_code = 'java/code/path/here' % (can just bypass this if not using java, and replace with your own policy evaluation call. See comments below.)

% Domain-related settings
grid_size = 6; %grid size
probNum = -1;
flat = false;
numHoriz = 25; % horizon length
discount = 0.99; % discount factor
has_converged = false;

trial = 3

if trial == 1
    % trial - default G-DICE with alpha 0.5
    shouldUseAdaptiveLearning = 0;
    shouldUpdateWorstEliteSample = 1;
    shouldInjectNoise = 0; 
    shouldInjectNoiseUsingMaximalEntropy = 0;
    shouldUseMealy = 1;
    
    if shouldInjectNoise && ~shouldInjectNoiseUsingMaximalEntropy
        % for naive noise injection, always inject
        shouldInjectNoiseNow = true;
    else
        % for maxEnt noise injection (or no injection), begin with no noise
        shouldInjectNoiseNow = false;
    end
    
    alpha = 0.5; %learning rate
elseif trial == 2
    % trial - adaptive learning, b=0.5 q=15
    shouldUseAdaptiveLearning = 1;
    shouldUpdateWorstEliteSample = 0;
    shouldInjectNoise = 0; % TODO fix this logic
    shouldInjectNoiseUsingMaximalEntropy = 1; % ==0 fixed noise injection, ==1 maximal entropy noise injection
    shouldUseMealy = 1;
    
    if shouldInjectNoise && ~shouldInjectNoiseUsingMaximalEntropy
        % for naive noise injection, always inject
        shouldInjectNoiseNow = true;
    else
        % for maxEnt noise injection (or no injection), begin with no noise
        shouldInjectNoiseNow = false;
    end
    
    alpha = 0.5; %learning rate
    if shouldUseAdaptiveLearning
        beta = alpha; % adaptive rate parameter
        q = 15; % adaptive rate parameter
    end
elseif trial == 3
    % trial - CE injection w/ maxEntropyMonitoring
    shouldUseAdaptiveLearning = 0;
    shouldUpdateWorstEliteSample = 1;
    shouldInjectNoise = 1; % TODO fix this logic
    shouldInjectNoiseUsingMaximalEntropy = 1; % ==0 fixed noise injection, ==1 maximal entropy noise injection
    shouldUseMealy = 1;
    
    if shouldInjectNoise && ~shouldInjectNoiseUsingMaximalEntropy
        % for naive noise injection, always inject
        shouldInjectNoiseNow = true;
    else
        % for maxEnt noise injection (or no injection), begin with no noise
        shouldInjectNoiseNow = false;
    end
    
    alpha = 0.5; %learning rate
end

% Graph controller-related settings (these are also domain-related)
numNodes = 5; %number of graph controller nodes
numAgents = 2;
numTmas = [4 4]; % number of TMAs for each agent
numObs = 95; % cardinality of observation space (7 states, 3 health states)
N_k = 600; % number of iterations
N_s = 100; % number of samples per iteration
N_b = 5; % number of "best" samples kept from each iteration

% Java code link
javaaddpath(path_java_code)
% This corresponded to the main function of our domain code (which was written in Java) and used for evaluation of policies only. This code is not needed for general domains.
o = ProblemSetupFactor(probNum,grid_size,numAgents,flat); 

% Output folder
foldername = ['output\_bestresults_gdice_' datestr(now,'yyyy_mm_dd_HH_MM_SS') '\'];
if ~exist(foldername, 'dir')
    mkdir(foldername)
end

for idxAgent = 1:numAgents
    if shouldUseMealy
        mGraphPolicyController(idxAgent) = GraphPolicyController_mealy(numNodes, alpha, numTmas(idxAgent), numObs, N_s, shouldUseAdaptiveLearning, shouldUpdateWorstEliteSample, shouldInjectNoise, shouldInjectNoiseUsingMaximalEntropy);
    else
        mGraphPolicyController(idxAgent) = GraphPolicyController(numNodes, alpha, numTmas(idxAgent), numObs, N_s);
    end
end

worstEliteValue = -100;
minValueAllowed = -100;
bestValue = minValueAllowed;
allValues = zeros(N_k*N_s,1);

fHandle = figure;

for idxIteration = 1:N_k
    if (bestValue>minValueAllowed) 
        if (shouldUpdateWorstEliteSample)
            minValueAllowed = worstEliteValue
        else
            minValueAllowed = -1000
        end
    end
    if shouldUseAdaptiveLearning
        mGraphPolicyController(1).alpha = beta - beta*(1-1/idxIteration)^q;
        mGraphPolicyController(2).alpha = beta - beta*(1-1/idxIteration)^q;
        alpha = mGraphPolicyController.alpha;
    end
        
    %reset N_b best sample values
    curIterationValues = -100*ones(N_s,1); % Set to a large negative number so we can track best values
    %draw N_s samples based on pdfs
    for idxAgent = 1:numAgents
        mGraphPolicyController(idxAgent).sample(N_s);
    end
    %evaluate each sample
    for idxSample = 1:N_s
        fprintf('Iteration %d of %d. Best value so far: %f\n', (idxIteration-1)*N_s + idxSample, N_k*N_s, bestValue);
        
        for idxAgent = 1:numAgents
            mGraphPolicyController(idxAgent).setGraph(idxSample);
        end
        [bestTMAs1, bestTransitions1] = mGraphPolicyController(1).getPseudoMealyPolicyTable();
        [bestTMAs2, bestTransitions2] = mGraphPolicyController(2).getPseudoMealyPolicyTable();
        
        % convert deterministic Mealy table to a stochastic one (put
        % probability 1 on the chosen MA, and 0 on the remaining)
        featureToAct = zeros(numAgents,numObs,numNodes,numTmas(1));
        featureTrans = zeros(numAgents, numTmas(1), numObs, numNodes, numNodes);
        
        for idx_agent = 1:numAgents
            for idx_obs = 1:numObs
                for idx_node = 1:numNodes
                    if idx_agent == 1
                        featureToAct(1,idx_obs,idx_node,bestTMAs1(idx_node,idx_obs)) = 1;
                    else
                        featureToAct(2,idx_obs,idx_node,bestTMAs2(idx_node,idx_obs)) = 1; 
                    end
                end
            end
        end
        
        % converting policy tables to java domain format (this is domain-specific, so can be removed for other domains)
        for idx_node = 1:numNodes
            for idx_agent = 1:numAgents
                for idx_action = 1:numTmas(1)
                    featureTrans(idx_agent,idx_action,:,idx_node,:) = permute(reshape(mGraphPolicyController(idx_agent).nodes(idx_node).pTableNextNode,[1, 1, numNodes, 1, numObs]),[1 2 5 4 3]);
                end
            end
        end

        proba_start_nodes = zeros(numNodes,1);
        proba_start_nodes(1) = 1;
        
        % This is where policy evaluation was done using our java method (can replace with your own evaluation function call)
        mean_reward = javaMethod('main', o, numNodes, grid_size, numHoriz, discount, proba_start_nodes, featureToAct, featureTrans);
        curIterationValues(idxSample) = mean_reward;
    end
            
    curIterationValues
    
    for idxSample = 1:N_s
        allValues((idxIteration-1)*N_s + idxSample) = curIterationValues(idxSample);
        
        if (curIterationValues(idxSample)> bestValue)
            bestValue = curIterationValues(idxSample);
            for idxAgent = 1:numAgents
                mGraphPolicyController(idxAgent).setGraph(idxSample);
            end
            [bestTMAs1, bestTransitions1] = mGraphPolicyController(1).getPseudoMealyPolicyTable();
            [bestTMAs2, bestTransitions2] = mGraphPolicyController(2).getPseudoMealyPolicyTable();
            
            filename = [foldername num2str((idxIteration-1)*N_s + idxSample) '_value_' num2str(mean(curIterationValues)) '.mat'];
            
            fprintf('Saving results with bestValue: %f to file %s\n', bestValue);
            filename
            idxSample
            
            save(filename,'bestTMAs1','bestTMAs2', 'bestTransitions1', 'bestTransitions2','allValues','numNodes','probNum','flat','numHoriz','discount','alpha','numTmas','numObs','N_k','N_s','N_b','numAgents','shouldUseAdaptiveLearning','shouldUpdateWorstEliteSample','shouldInjectNoise','shouldUseMealy','grid_size','mGraphPolicyController')
        end
    end
    
    %performs update and filter of pdfs
    just_injected_noise = false;
    for idxAgent = 1:numAgents
        [worstEliteValue, maxValueIdxs, entropy_sums, just_injected_noise] = mGraphPolicyController(idxAgent).updateProbs(curIterationValues, N_b, minValueAllowed, idxIteration, shouldInjectNoiseNow);
    end
    
    
    if just_injected_noise
        worstEliteValue = -100;
    end
    
    [sortedValues, sortingIndices] = sort(curIterationValues,'descend');
    maxValues = sortedValues(1:N_b)
    maxValueIdxs = sortingIndices(1:N_b);
    maxValueIdxs = maxValueIdxs(maxValues>minValueAllowed);
    curIterationValues(maxValueIdxs)
    pause(0.0001) 
end