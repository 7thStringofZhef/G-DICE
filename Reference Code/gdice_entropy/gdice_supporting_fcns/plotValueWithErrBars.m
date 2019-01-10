function [mHandle,valAvgs] = plotValueWithErrBars(fHandle, data, N_s, color)
    %inputs: data (length(x)-by-numSamples size)
    figure(fHandle)
    numIters = length(data)/N_s;
    valAvgs = zeros(numIters,1);
    valStDevs = zeros(numIters,1);
    
    for idxIteration = 1:numIters
        samplesStart = (idxIteration-1)*N_s+1;
        samplesEnd = (idxIteration)*N_s;
        valAvgs(idxIteration) = mean(data(samplesStart:samplesEnd));
        valStDevs(idxIteration) = std(data(samplesStart:samplesEnd));
    end
    
    mHandle = shadedErrorBar(1:numIters,valAvgs,valStDevs,{'color',color},1);
    mHandle = mHandle.mainLine;
    hold on
end