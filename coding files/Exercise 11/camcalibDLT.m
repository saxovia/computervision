function P = camcalibDLT(worldPoints, imagePoints)

    nPoints = size(worldPoints, 1);
    A = zeros(2*nPoints, 12);

    for i = 1:nPoints
        X = worldPoints(i,:); % homogenous
        x = imagePoints(i,1);
        y = imagePoints(i,2);

        A(2*i-1,:) = [0, 0, 0, 0, -X, y*X];
        A(2*i,:) = [X, 0, 0, 0, 0, -x*X];
    end

    % minimize ||Ap||^2
    % svd
    [~, ~, V] = svd(A);
    P = reshape(V(:, end), 4, 3)';
end
