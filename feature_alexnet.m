rootFolder = fullfile('G:\My Drive\Internship\Machine Learning\training', 'Task1');
imgSets = imageSet('G:\My Drive\Internship\Machine Learning\training\Task1', 'recursive');
% rootFolder = fullfile('C:\Users\umarh\Downloads\Images\', 't1_testing_images');
% imgSets = imageSet('C:\Users\umarh\Downloads\Images\t1_testing_images', 'recursive');

n=length(imgSets);
courseNames = cell(n,1);  
   % pre-size classNames as a row vector
 for k=1:n
          courseNames{k} = imgSets(k).Description;
 end
 % convert courseNames to a character matrix
 categories=courseNames;

%IMDB
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames'); 
tbl = countEachLabel(imds);


  minSetCount = min(tbl{:,2});

 imds = splitEachLabel(imds,minSetCount, 'randomize');

pretrainmodel = alexnet
 
convnet=load('E:\acadamic\MS (CS)\project\coding\cnn\vg.mat');
convnet=convnet.convn;

imds.ReadFcn = @(filename)ImageProcesing(filename);
[trainingSet, testSet] = splitEachLabel(imds, 0.05, 'randomize');
%--------]




featureLayer = 'fc6';
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
'MiniBatchSize', 20, 'OutputAs', 'rows');
trainingLabels = trainingSet.Labels;
%classifier  
  classifier = fitcecoc(trainingFeatures, trainingLabels, ...
  'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');
%classifier =fitcdiscr(trainingFeatures, trainingLabels);
%  classifier = fitcdiscr(trainingFeatures, trainingLabels);%LDA classifier
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',20, 'OutputAs', 'rows');
predictedLabels = predict(classifier, testFeatures);
testLabels = testSet.Labels;
accuracy = mean(predictedLabels == testLabels);

% [confMat,order] = confusionmat(testLabels,predictedLabels)
% for i =1:size(confMat,1)
%     recall(i,:)=(confMat(i,i)/sum(confMat(i,:)));
% end
% recall(isnan(recall))=[];
% 
% Recall=sum(recall)/size(confMat,1);
% for i =1:size(confMat,1)
%     precision(i,:)=confMat(i,i)/sum(confMat(:,i));
% end
% Precision=sum(precision)/size(confMat,1);
% 
% F_score=2*Recall*Precision/(Precision+Recall); 



% end

% I=imread('C:\Users\umarh\OneDrive\Desktop\tracks_cropped\00068.jpg'); 
% 
%         if ismatrix(I)
%             I = cat(3,I,I,I);
%         end
% 
%         % Resize the image as required for the CNN.
%         I = imresize(I, [227 227]);
%         
%         
%         testFeatures = activations(convnet, I, featureLayer, 'MiniBatchSize',20, 'OutputAs', 'rows');
%  predictedLabels = predict(classifier, testFeatures)
% 
% %fclose(fileID);