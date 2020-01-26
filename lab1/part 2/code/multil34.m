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
 
% _______________________________4.3.1______________________________________ 
%% Two-layer perceptron for time series predicition 

% Choose a Training Function (help nntrain)
trainFcn = 'trainscg'; % Variable Learning Rate Gradient Descent

% Early stopping
%net.trainParam.max_fail = 6;    % Maximum validation failures

   % Create a Fitting Network 
        net = fitnet(8,trainFcn);
        
        % Set strength of regularization 
        net.performParam.regularization = 0;
        
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
        
            % Initalize
            net = init(net);
            

            t_noise = t + normrnd(0, 0.09, [1,1200]);

            time_i = cputime;
            % Train the Network
            [net,tr] = train(net,x,t_noise);           
            time_e = cputime-time_i;

      time_e     
   




