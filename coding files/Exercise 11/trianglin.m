function X = trianglin(P1, P2, x1, x2)


    A = [ x1(1)*P1(3,:) - P1(1,:);
          x1(2)*P1(3,:) - P1(2,:);
          x2(1)*P2(3,:) - P2(1,:);
          x2(2)*P2(3,:) - P2(2,:);   ];

    % svd -> minimize ||A*X||
    [~, ~, V] = svd(A);
    X = V(:, end);
    % normalization
    X = X / X(end);
end
