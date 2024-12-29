function  F  = estimateF(points1,points2)

N=size(points1,1);
F = zeros(N,9);
for i=1:N
    x1=points1(i,1); y1=points1(i,2); w1=points1(i,3);
    x2=points2(i,1); y2=points2(i,2); w2=points2(i,3);
    F(i,:)=[x2*x1, x2*y1, x2*w1, y2*x1, y2*y1, y2*w1, w2*x1, w2*y1, w2*w1];
end
[~, ~, V] = svd(F);
F= reshape(V(:,end),[3,3])';
[U,S,V]=svd(F);
S(3,3)=0;
F=U*S*V'

end

