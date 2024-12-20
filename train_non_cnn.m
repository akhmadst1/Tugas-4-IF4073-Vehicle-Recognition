% Load the dataset
datasetPath = './dataset';
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display the distribution of labels
disp(countEachLabel(imds));

% Split the data into training and testing sets
[trainImds, testImds] = splitEachLabel(imds, 0.8, 'randomized');

% Initialize variables
trainFeatures = [];
trainLabels = [];
testFeatures = [];
testLabels = [];

% Extract HOG features for training data
for i = 1:numel(trainImds.Files)
    img = readimage(trainImds, i);
    imgGray = rgb2gray(img); % Convert to grayscale
    imgResized = imresize(imgGray, [64 64]); % Resize images
    hogFeatures = extractHOGFeatures(imgResized);
    trainFeatures = [trainFeatures; hogFeatures];
    trainLabels = [trainLabels; trainImds.Labels(i)];
end

% Extract HOG features for testing data
for i = 1:numel(testImds.Files)
    img = readimage(testImds, i);
    imgGray = rgb2gray(img); % Convert to grayscale
    imgResized = imresize(imgGray, [64 64]); % Resize images
    hogFeatures = extractHOGFeatures(imgResized);
    testFeatures = [testFeatures; hogFeatures];
    testLabels = [testLabels; testImds.Labels(i)];
end

% Train an SVM classifier
svmModel = fitcecoc(trainFeatures, trainLabels);

% Test the model
predictedLabels = predict(svmModel, testFeatures);

% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

% Save the trained SVM model
save('svmModel.mat', 'svmModel');
disp('SVM model saved to svmModel.mat');