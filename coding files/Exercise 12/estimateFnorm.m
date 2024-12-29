function Fnorm=estimateFnorm(points1,points2)
    [normalizedpoint1,T1]=normalizePoints(points1);
    [normalizedpoint2,T2]=normalizePoints(points2);
    
    F=estimateF(normalizedpoint1,normalizedpoint2);
    Fnorm=T2'*F*T1;
end


function [normalizedpoint,T]=normalizePoints(points)
    origin=mean(points(:,1:2),1);
    
    % shift to origin
    shiftedpoints=points;
    shiftedpoints(:,1)=points(:,1)-origin(1);
    shiftedpoints(:,2)=points(:,2)-origin(2);

    distances=sqrt(sum(shiftedpoints(:,1:2).^2,2));
    meandist=mean(distances);
    
    scale=sqrt(2)/meandist;
    T=[scale 0 -scale*origin(1);
       0 scale -scale*origin(2);
       0 0 1];
    
    normalizedpoint=(T*points')';
end