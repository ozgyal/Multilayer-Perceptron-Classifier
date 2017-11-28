% Prepare train and test sets
clc
clear

% Read train set
fid = fopen('ORL_tr.txt','r');
formatSpec = '%d,';
train_set = zeros(200,256);
train_labels = zeros(200,1);
imCount = 1;
while imCount ~= 201
    image = fscanf(fid,formatSpec);
    image = image';
    train_set(imCount,:) = image(1,1:256);
    train_labels(imCount,1) = image(1,257);
    imCount = imCount+1;
end
fclose(fid);

% Read test set
fid = fopen('ORL_ts.txt','r');
formatSpec = '%d,';
test_set = zeros(200,256);
test_labels = zeros(200,1);
imCount = 1;
while imCount ~= 201
    image = fscanf(fid,formatSpec);
    image = image';
    test_set(imCount,:) = image(1,1:256);
    test_labels(imCount,1) = image(1,257);
    imCount = imCount+1;
end
fclose(fid);

% Normalize train dataset
avg = mean(train_set);
for i=1:256
    temp(:,i) = (train_set(:,i) - avg(i)).^2;
end
deviation = sqrt((sum(temp)./200));

for i=1:256
    normalizedTrain(:,i) = (train_set(:,i) - avg(i))./deviation(i);
end

% Normalize test dataset
for i=1:256
    normalizedTest(:,i) = (test_set(:,i) - avg(i))./deviation(i);
end