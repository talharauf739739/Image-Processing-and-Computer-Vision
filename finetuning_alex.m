rootFolder = fullfile('G:\My Drive\Internship\Machine Learning\training', 'Task1');
imgSets = imageSet('G:\My Drive\Internship\Machine Learning\training\Task1', 'recursive');
n=length(imgSets);
courseNames = cell(n,1);  
   % pre-size classNames as a row vector
 for k=1:n
          courseNames{k} = imgSets(k).Description;
 end
 categories=courseNames;
%IMDB
for lop=1:1
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames'); 
tbl = countEachLabel(imds);
 minSetCount = min(tbl{:,2});
 imds = splitEachLabel(imds,minSetCount, 'randomize');

%convnet=load('E:\acadamic\MS (CS)\project\coding\cnn\alex_var.mat');
convnet=convnet.convn;
 imds.ReadFcn = @(filename)ImageProcesing(filename);
 [trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');
 layersTransfer = convnet.Layers(1:end-3);
numClasses =numel(categories);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MiniBatchSize',20, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-2, ...
    'Verbose',true, ...
    'Plots','training-progress');
 netTransfer = trainNetwork(trainingSet,layers,options);
YPred = classify(netTransfer,testSet);
accu= mean(YPred == testSet.Labels);
Conf_Mat = confusionmat(testSet.Labels,YPred);
disp(Conf_Mat)
confusionchart(gather(testSet.Labels),gather(YPred))
end

