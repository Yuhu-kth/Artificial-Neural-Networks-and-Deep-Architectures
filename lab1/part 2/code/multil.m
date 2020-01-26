% _______________________________4.3.0______________________________________ 
%% Generate data 
close all
clear all
clc
% data
d = 1.5;
 
for i = 1:1600
    if i-1<25
        d(end+1) = 0.9*d(i);
    else
        d(end+1) = 0.9*d(i) + 0.2*d(i-25)/(1+(d(i-25))^10);
    end
end
 
m = 301:1500;
x = [d(m-20); d(m-15); d(m-10); d(m-5); d(m)];
t = d(m+5);
 
plot(d)
title('Mackey-Glass time series')
xlabel('t')
ylabel('x(t)')
axis([301 1500 0 2])


% _______________________________4.3.1______________________________________ 
%% Two-layer perceptron for time series predicition 

% Choose a Training Function (help nntrain)
trainFcn = 'trainscg'; 

% Early stopping
%net.trainParam.max_fail = 6;    % Maximum validation failures

% Configurations
reg_strength = [0.0, 0.1, 0.5, 1.0];
hidden = [2, 4, 6, 8];

% Calculate average mse and std for each configuration 
train_mse_matrix = zeros(length(reg_strength), length(hidden));
val_mse_matrix = zeros(length(reg_strength), length(hidden));
test_mse_matrix = zeros(length(reg_strength), length(hidden));

best_test = 1000;

for r = 1:length(reg_strength)
    for h = 1:length(hidden) 
        weights = [];
        % Set number of hidden nodes
        hiddenLayerSize = hidden(h);

        % Create a Fitting Network 
        net = fitnet(hiddenLayerSize,trainFcn);
        
        % Set strength of regularization 
        net.performParam.regularization = reg_strength(r);
        
        % Setup Division of Data for Training, Validation, Testing
        net.divideFcn = 'divideind'; % Divide targets into three sets using specified indices
        net.divideParam.trainInd = 1:800; 
        net.divideParam.valInd = 801:1000;
        net.divideParam.testInd = 1001:1200;

        % Choose a Performance Function (help nnperformance)
        net.performFcn = 'mse';  % Mean Squared Error

        % Choose Plot Functions (help nnplot)
        net.plotFcns = {'plotperform', 'plotfit'};
        
        
        
        
        
        % Initialize mse for this configuration
        train_mse = [];
        val_mse = [];
        test_mse = [];
        
          repeats = 100;
        for i = 1:repeats
            % Initalize
            net = init(net);
            

            % Train the Network
            [net,tr] = train(net,x,t);

            % Test the Network
            y = net(x);
            e = gsubtract(t,y);
            %performance = perform(net,t,y);
            nntraintool close;

            % Recalculate Training, Validation and Test Performance
            trainTargets = t .* tr.trainMask{1};
            trainPerformance = perform(net,trainTargets,y);
            train_mse = [train_mse trainPerformance];

            valTargets = t .* tr.valMask{1};
            valPerformance = perform(net,valTargets,y);
            val_mse = [val_mse valPerformance];

            testTargets = t .* tr.testMask{1};
            testPerformance = perform(net,testTargets,y);
            test_mse = [test_mse testPerformance];
            
            weights = [weights; net.iw{1,1}];
            
        end
  
        train_mse_matrix(r,h) = mean(train_mse);
        val_mse_matrix(r,h) = mean(val_mse);
        test_mse_matrix(r,h) = mean(test_mse);  
        
        
        % Plot a test prediction for an example model of better configuration
        if val_mse_matrix(r,h) < best_test
            figure
            hold on
            plot(linspace(1301, 1500, 200), t(1001:1200));
            plot(linspace(1301, 1500, 200), y(1001:1200));
            title({['Selected model']; ['Number of hidden nodes: ' num2str(hidden(h))]; ['Strength of regularisation: ' num2str(reg_strength(r))]})
            ylabel('x(t+5)')
            xlabel('t')
            legend('True target values','Test predictions')
            best_test = val_mse_matrix(r,h);

        end
        
    end
    
    figure
    histogram(weights, 'BinWidth', 0.25)
    title({['Histogram of weights']; ['Strength of regularisation: ' num2str(reg_strength(r))];['Number of hidden nodes: ' num2str(hidden(h))]});
    xlabel('Weight value')
    ylabel('Counts')
end

val_mse_matrix
test_mse_matrix





