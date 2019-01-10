function entropy_vector = entropy_columnwise(pTable,nCols)
    % save time by passing in nCols instead of using size(pTable)
    entropy_vector = zeros(1,nCols);
    for idx_col = 1:nCols
        pCol = pTable(:,idx_col);
        entropy_vector(idx_col) = sum(-(pCol(pCol>0).*(log2(pCol(pCol>0)))));
    end
end