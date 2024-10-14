load('points.mat','x','y');
figure;hold on;
plot(x,y,'kx');
axis equal




N = 500;
threshold = 5;
max = 0;
bestline = [0,0];

for i = 1:N
    ind = randperm(length(x), 2);
    firstind = ind(1);
    secondind = ind(2);

    x1 = x(firstind);
    y1 = y(firstind);
    x2 = x(secondind);
    y2 = y(secondind);

    p= polyfit([x1,x2],[y1,y2],1);
    m = p(1)
    b= p(2)
    
    count = 1;
    for j = 1:length(x)
        if j ~= firstind && j ~= secondind
            xp = x(j);
            yp = y(j);
            dist = abs(-m*xp + yp - b) / sqrt(1 + m^2);
            if dist <= threshold
                count = count + 1;
            end
        end
    end
    
    if count > max
        max = count;
        bestline = [m,b];
    end

end

fprintf('Inliers: %d\n', max);
fprintf('Best line coeffs: m= %f, b=%f\n', bestline(1), bestline(2));

%-------
m = bestline(1);
b = bestline(2);
xcoords = [];
ycoords = [];

for i = 1:length(x)
    xi = x(i);
    yi = y(i);
    dist = abs(-m * xi + yi - b) / sqrt(1 + m^2);
    if dist <= threshold
        xcoords(end+1) = xi;
        ycoords(end+1) = yi;
    end
end

p_refit = polyfit(xcoords, ycoords, 1);
m_refit = p_refit(1);
b_refit = p_refit(2);

figure;
plot(x, y, 'kx'); 
hold on;
axis equal;
plot(x, bestline(1) .* x + bestline(2), 'b', 'DisplayName', 'First fit');

plot(x, m_refit .* x + b_refit, 'r', 'DisplayName', 'Refitted line');
legend('Data points', 'RANSAC', 'Refitted line');