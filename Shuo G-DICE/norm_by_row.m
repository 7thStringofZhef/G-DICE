function new_matrix=norm_by_row(matrix)
[m,n]=size(matrix);
new_matrix=matrix;
for i=1:m
    if(sum(new_matrix(i,:))~=0)
        new_matrix(i,:)=new_matrix(i,:)/sum(new_matrix(i,:));
    end
end
end