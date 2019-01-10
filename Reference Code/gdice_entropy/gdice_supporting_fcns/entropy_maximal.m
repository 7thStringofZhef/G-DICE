function max_entropy = entropy_maximal(nRows)
    pVec = ones(nRows,1)/nRows;
    max_entropy = sum(-(pVec(pVec>0).*(log2(pVec(pVec>0)))));
end