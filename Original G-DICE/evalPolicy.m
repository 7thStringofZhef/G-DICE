function [finalValue, completionTime, successProb] = evalPolicy(policyController, numPackageGoal, isOutputOn, useJointTMAs)
    maxRuns = 80;
    values = zeros(maxRuns,1);
    completionTime = zeros(maxRuns,1);
    
%     textprogressbar('calculating outputs: ');
    for idxRun = 1:maxRuns
        if (nargin == 4)
            dom = DomainNoJointTMAs();
        else
            dom = Domain();
        end
        
        dom.setPolicyForAllAgents(policyController);
        
        if (nargin == 1)
            isOutputOn = false;
            values(idxRun) = dom.evalCurPolicy(isOutputOn);
        elseif (nargin == 2)
            isOutputOn = false;
            [values(idxRun) completionTime(idxRun)] = dom.evalCurPolicy(isOutputOn, numPackageGoal);
        else
            if (~isempty(numPackageGoal)) 
                [values(idxRun) completionTime(idxRun)] = dom.evalCurPolicy(isOutputOn, numPackageGoal);
            else
                values(idxRun) = dom.evalCurPolicy(isOutputOn);
            end
        end
%         textprogressbar((idxRun/maxRuns)*100);     
    end
%     textprogressbar('done');
    
    fprintf('Avg value %f\n', mean(values))
    fprintf('Best value %f\n\n', max(values))
    
    
    
    %     valuePlotX = 1:length(values);
    valuePlotY = zeros(1,length(values));
    
    for idxValue = 1:length(values)
        valuePlotY(idxValue) = sum(values(1:idxValue))/idxValue;
    end
    
    finalValue = valuePlotY(end);
    
    %average completion times for completed policies (incomplete policies have competionTime == -1)
    successProb = length(completionTime(completionTime~=-1))/length(completionTime);
    completionTime = sum(completionTime(completionTime~=-1))/length(completionTime(completionTime~=-1));
    
    %     plot(valuePlotX, valuePlotY)
    %     grid on
    %     xlabel('Run #')
    %     ylabel('Policy value (running avg)')
    %     axis([0 length(values) 0 max(valuePlotY)*1.2])
    %
    %     set(gcf,'PaperPositionMode','auto')
    %     print('-dpng','-zbuffer','-r200','policyValueConvg.png')
end