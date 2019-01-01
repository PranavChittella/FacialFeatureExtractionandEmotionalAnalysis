

%This script tests the pre-selected test images, which is cropped in
%EE368_FinalProject_CropFaces.m and classifies them. 

%track execution runtime
tic;

%Object detectors for eyes, nose, and mouth
eyesDetector = vision.CascadeObjectDetector('EyePairBig');
mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 15);

if (exist('EmotionFisherFaces') == 0)
    fprintf('Please first perform training on data using EE368_FinalProject_Train\n')
else
    fprintf('Woohoo!\n')
    
    TestImages = 'TestFacesCrop';
    Testing_Folder = dir(fullfile(TestImages, '*.png'));
    
    for k = 1:length(Testing_Folder)
        
        
        ImTest = im2double(imread(fullfile(TestImages, Testing_Folder(k).name)));  % Read image
        ImTest = reshape(ImTest, [100^2, 1]);
        
        %Binary prediction array to hold Fisherface predictions in the
        %order: 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise'.
        Predictions = zeros(1,5);
        
        for j = 1:length(EmotionFisherFaces(:,1))
            
            fisherface = EmotionFisherFaces(j,:);
            Pcoeff = dot(fisherface, ImTest);
            
            %Experimentally determined thresholds or the 'easy to
            %determine' expressions
            %
            if j == 1
                %Anger
                if Pcoeff > 1
                    Predictions(1) = 1;
                end
            elseif j == 2
                %Fear
                if Pcoeff > 1
                    Predictions(2) = 1;
                end
            elseif j == 3
                %Happy
                if Pcoeff < -1
                    Predictions(3) = 1;
                end
            elseif j == 4
                %Sad
                if Pcoeff < 0.6
                    Predictions(4) = 1;
                end
            elseif j == 5
                %Surprise
                if Pcoeff < -0.5
                    Predictions(5) = 1;
                end
            end
        end
        
        PredictedEmotion = 'None';
        
        %If one of the five 'easy-to-distinguish' is singularly predicted
        %(no other positive predictions), make that the emotion prediction.
        if Predictions(1) == 1 && Predictions(2) == 0 && ...
                Predictions(3) == 0 && Predictions(4) == 0 && Predictions(5) == 0
            PredictedEmotion = 'Anger';
        elseif Predictions(1) == 0 && Predictions(2) == 1 && ...
                Predictions(3) == 0 && Predictions(4) == 0 && Predictions(5) == 0
            PredictedEmotion = 'Fear';
        elseif Predictions(1) == 0 && Predictions(2) == 0 && ...
                Predictions(3) == 1 && Predictions(4) == 0 && Predictions(5) == 0
            PredictedEmotion = 'Happy';
        elseif Predictions(1) == 0 && Predictions(2) == 0 && ...
                Predictions(3) == 0 && Predictions(4) == 1 && Predictions(5) == 0
            PredictedEmotion = 'Sad';
        elseif Predictions(1) == 0 && Predictions(2) == 0 && ...
                Predictions(3) == 0 && Predictions(4) == 0 && Predictions(5) == 1
            PredictedEmotion = 'Surprise';
        else
            fprintf('Further predicitive measures needed\nProceeding to HOG extraction\n');
            
            
            EmotionFace = reshape(ImTest, [100 100]);
           
            %find features in each image
            eyeFind = step(eyesDetector, EmotionFace);
            foundEyesIm = insertObjectAnnotation(EmotionFace, 'rectangle', eyeFind, 'Eyes');
            
            
            
            if isempty(eyeFind)
                eyeBox = FindEyesWithKeypoints(EmotionFace);
            else
                rowE = eyeFind(2);
                colE = eyeFind(1);
                colAddE = eyeFind(3);
                rowAddE = eyeFind(4);
                
                centerColE = round(colE + colAddE/2);
                
                eyeWidth = 70;
                eyeHeight = 30;
                eyeBox = EmotionFace(rowE-10:rowE+eyeHeight-1, centerColE - eyeWidth/2 : centerColE + eyeWidth/2);
            end
            
            [featureVectorE,hogVisualizationE] = extractHOGFeatures(eyeBox,'CellSize',[4 4]);
            
            %Show extracted eye box with extracted HOG features
            %imshow(eyeBox);
            %hold on;
            %plot(hogVisualizationE);
            
            
            rowSize = length(EmotionFace(:,1));
            colSize = length(EmotionFace(1,:));
            
            
            lookForMouth = EmotionFace;
            lookForMouth(1:rowSize*0.75,:) = 0;
            
            mouthFind = step(mouthDetector, lookForMouth);
            foundMouthIm = insertObjectAnnotation(lookForMouth, 'rectangle', mouthFind, 'Mouth');
            
            mouthBox = EmotionFace(end-30: end, 50 - 20: 50 + 20);
            
            
            [featureVectorM,hogVisualizationM] = extractHOGFeatures(mouthBox,'CellSize',[4 4]);
            HOGvector = horzcat(featureVectorE, featureVectorM);
            
            [labelPred, scores] = predict(MCSVM, HOGvector);
            
            if strcmp(labelPred, 'A')
                PredictedEmotion = 'Angry';
            elseif strcmp(labelPred, 'C')
                PredictedEmotion = 'Contempt';
            elseif strcmp(labelPred, 'D')
                PredictedEmotion = 'Disgust';
            elseif strcmp(labelPred, 'F')
                PredictedEmotion = 'Fear';
            elseif strcmp(labelPred, 'H')
                PredictedEmotion = 'Happy';
            elseif strcmp(labelPred, 'S')
                PredictedEmotion = 'Sad';
            elseif strcmp(labelPred, 'O')
                PredictedEmotion = 'Surprise';
            end
%             
%             
        end
        %
%         figure
        %show the test image and the accompanying prediction
        figure,imshow(reshape(ImTest, [100 100]))
        title(strcat('Prediction:', '"', PredictedEmotion,'"'));
        pause(0.5)
        
    end
    
end


%track execution runtime
TimeSpent = toc