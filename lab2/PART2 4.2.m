% 4.2
clear all
close all
% import data
data = importdata('cities.dat');
data = data(3:end,:);%从第三行开始，前两行是注释
%之前是cell型数组，最终转换为num
data = char(data);
coordinates = str2num(data);
shuffled_idx = randperm(size(coordinates,1));
coordinates = coordinates(shuffled_idx,:);
shuffled_idx
coordinates
% initialise weights 
W = rand(2,10);%返回在区间 (0,1) 内均匀分布的随机数

% SOM 
n_epochs = 20;
step_size = 0.2;
 
neighbourhood(1,1:n_epochs) = 2;
for i=2:length(neighbourhood)
    neighbourhood(i) = neighbourhood(i-1)*0.8;
end
neighbourhood = round(neighbourhood);

%{
stepsize(1,1:n_epochs) = 0.2;
for i=2:length(stepsize)
    stepsize(i) = stepsize(i-1)*0.65;
end
stepsize
%衰减step size
%}

for epoch=1:n_epochs
    % start with large neighbourhood then decrease
    % neighbour reach indices
    n_reach = neighbourhood(epoch);
%step_size = stepsize(epoch)
    % train for each cities
    for i=1:10
        % calculate winner
        distance = sum((W-coordinates(i,:)').^2,1);
        [~,winner] = min(distance);
        

        % find neighbours
        max_n = winner + n_reach;
        min_n = winner - n_reach;
        neighbours = zeros(1,10);
        if(max_n > 10)
            neighbours(winner:10) = 1;
            neighbours(1:max_n-10) = 1;
        else
            neighbours(winner:max_n) = 1;
        end
        if(min_n < 1)
            neighbours(1:max_n) = 1;
            neighbours(10+min_n:10) = 1;
        else
            neighbours(min_n:winner) = 1;
        end
           
        
        % update weights
        for j=1:10
            % if neighbour, update weights
            if(neighbours(j) == 1)
                W(:,j) = W(:,j) + step_size.*(coordinates(i,:)' - W(:,j));   
            end
        end
    end
end

mask = ones(1,10);
% find winners
for i=1:10
    distance = sum((W-coordinates(i,:)').^2,1) .* mask;
    [~,winner] = min(distance);
    final_pos(i) = winner;
    mask(winner) = nan;
    
end
[~, idx] = unique(final_pos);
final_order = coordinates(idx,:);

% plot tour
figure
plot(final_order(:,1), final_order(:,2),'-o')
axis([0 1 0 1])
title({['Tour']; ['epochs: ' num2str(n_epochs)]; ['step size: ' num2str(step_size)]})
xlabel('x1')
ylabel('x2')
