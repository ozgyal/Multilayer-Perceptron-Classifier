clear all
close all
clc

prep_sets

train_set = normalizedTrain;
test_set = normalizedTest;
%% Determining the parameters

n = 64; % Number of hidden layers
c = 0.05; % The learning coefficient
max_epoch = 3000;
class_no = 40;
data_no = 200;

%% Initializing

% Initialize the weight matrices
w1 = random('unif',-0.05,0.05,n,257);
w2 = random('unif',-0.05,0.05,class_no,n+1);

% Initialize the activation function as bipolar sigmoidal function
f = @(v) (1-exp(-v))./(1+exp(-v));

% Initialize the derivative of bipolar sigmoidal function
fder = @(v) 0.5 .* ( 1 - f(v).^2 );

% Initialize the desired output(d)
d = (-1 * ones(40)) + (2 * eye(40));

% Initialize the train and test sets (just add -1 as threshold)
threshold = -1 * ones(data_no,1);
train_set = [train_set threshold];
train_set = train_set';
test_set = [test_set threshold];
test_set = test_set';

%% Training
for epoch=1:max_epoch
    Etotal=0;
    for i=1:data_no  % For each pattern 
        
        % First layer
        v1 = w1 * train_set(:,i);
        y = f(v1);
        y = [y; -1]; % Extended
        
        % Second layer    
        v2 = w2 * y;        
        o = f(v2);
        
        index = train_labels(i); % Returning the desired value corresponding to i
        % Find error
        e=d(:,index)-o;
        
        % Find local gradient for output layer
        deltaO = fder(v2) .* e;
        
        % Find local gradients for the first layer
        deltaF =  w2(:,1:n)' * deltaO .* fder(v1);
        
        % Update output layer weights
        w2 = w2 + c * deltaO * y';
        
        % Update first layer weights
        w1 = w1 + c * deltaF * train_set(:,i)';
        
        Etotal = Etotal + 0.5 * sum(e.^2); 
    end 

    Eave = Etotal / data_no; % Calculate the average error at the end of each epoch
    Eaves(epoch) = Eave; % Store the corresponding Eave for the plot

    if Eave <= 0.0001 %if stop condition is satisfied, stop training
        display0 = ['n = ',num2str(n),' c = ',num2str(c)];
        disp(display0)
        display1 = ['For Epoch:',num2str(epoch),' , stop condition is satisfied. Average error achieved is ',num2str(Eave),'.'];
        disp(display1)
        break;
    end
    if epoch == max_epoch % If the error is not less than 1e-4 for max_epoch, stop and display message, plot the graphs
        display0 = ['n = ',num2str(n), ' c = ',num2str(c)];
        disp(display0)
        display1 = ['Stop condition is not satisfied in ',num2str(max_epoch),' epochs. Average error for ',num2str(max_epoch),' epochs is ',num2str(Eave),'.'];
        disp(display1)
        break;
    end
    %plotting Eave throughout the epochs
    plot(Eaves,'LineWidth',4)
    axis([0 epoch 0 0.02])
    xlabel('Epoch')
    ylabel('Eave')
    title('Average error vs. # of Epoch');
end  

%% Calculate accuracy for train dataset
success = 0;
for i=1:data_no
    % First layer
    v1 = w1 * train_set(:,i);
    y = f(v1);
    y = [y; -1]; % Extended
        
    % Second layer    
    v2 = w2 * y;        
    o = f(v2);
    
    index = train_labels(i); % Returning the desired value corresponding to i   
    [~,maxO] = max(o);
    [~,maxD] = max(d(:,index));
    if maxO == maxD
       success = success + 1; 
    end
end
disp(['Accuracy for train set is: ' num2str((success * 100)/200) '%'])

%% Calculate accuracy for test dataset
success = 0;
for i=1:data_no
    % First layer
    v1 = w1 * test_set(:,i);
    y = f(v1);
    y = [y; -1]; % Extended
        
    % Second layer    
    v2 = w2 * y;        
    o = f(v2);
    
    index = test_labels(i); % Returning the desired value corresponding to i   
    [~,maxO] = max(o);
    [~,maxD] = max(d(:,index));
    if maxO == maxD
       success = success + 1; 
    end
end

disp(['Accuracy for test set is: ' num2str((success * 100)/200) '%'])