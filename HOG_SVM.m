
%Viola-Jones feature detection to extract eyes, nose, mouth

FaceFolders = {'AngryCrop', 'ContemptCrop', 'DisgustCrop', 'FearCrop', 'HappyCrop', 'SadCrop', 'SurpriseCrop'};



%Character codes:
% Angry -> 'A'
% Contempt -> 'C'
% Disgust -> 'D'
% Fear -> 'F'
% Happy -> 'H'
% Sad -> 'S'
% Surprise -> 'O'

FaceHOGs = zeros(1, 7128);
FaceLabels = 'X'; %non-label placeholder for character array

for i = 1:length(FaceFolders)
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    %Select an Emotion folder from the folder list (Angry, Contempt,
    %Disgust, Fear, Happy, Sad, Surprise)
    Emotion = FaceFolders{i};
    
    %read in all images
    TrainingImages = Emotion;
    Training_Folder = dir(fullfile(TrainingImages, '*.png'));
    
    
    for j = 1:length(Training_Folder)
        
        EmotionFace = im2double(imread(fullfile(TrainingImages, Training_Folder(j).name)));  % Read image
        
        %Object detectors for eyes, nose, and mouth
        eyesDetector = vision.CascadeObjectDetector('EyePairBig');
        noseDetector = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 30);
        mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 15);
        
        
        %find features in each image
        eyeFind = step(eyesDetector, EmotionFace);
        foundEyesIm = insertObjectAnnotation(EmotionFace, 'rectangle', eyeFind, 'Eyes');
        
        
        
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if ~isempty(eyeFind)
            rowE = eyeFind(2);
            colE = eyeFind(1);
            colAddE = eyeFind(3);
            rowAddE = eyeFind(4);
            
            centerColE = round(colE + colAddE/2);
            
            eyeWidth = 70;
            eyeHeight = 30;
            
            eyeBox = EmotionFace(rowE-10:rowE+eyeHeight-1, centerColE - eyeWidth/2 : centerColE + eyeWidth/2);
            [featureVectorE,hogVisualizationE] = extractHOGFeatures(eyeBox,'CellSize',[4 4]);
            
            %Show extracted eye box with extracted HOG features
%             imshow(eyeBox);
%             hold on;
%             plot(hogVisualizationE);
            
            
            rowSize = length(EmotionFace(:,1));
            colSize = length(EmotionFace(1,:));
            
            
            lookForMouth = EmotionFace;
            lookForMouth(1:rowSize*0.75,:) = 0;
            
            mouthFind = step(mouthDetector, lookForMouth);
            foundMouthIm = insertObjectAnnotation(lookForMouth, 'rectangle', mouthFind, 'Mouth');
            
            %             if ~isempty(mouthFind)
            %
            %                 rowM = mouthFind(1,2);
            %                 colM = mouthFind(1,1);
            %                 colAddM = mouthFind(1,3);
            %                 rowAddM = mouthFind(1,4);
            %
            %                 mouthCenter = round(colM + colAddM/2);
            %
            %                 mouthBox = EmotionFace(rowM - 10: rowM + 15, mouthCenter - 30: mouthCenter + 30);
            %
            %             else
            %                 lookForMouth(1:rowSize*0.75,:) = 1;
            %                 lookForMouth(:,1:25) = 1;
            %                 lookForMouth(:,end-25:end) = 1;
            %
            %
            %                 lookForMouth = im2bw(lookForMouth, 0.2);
            %                 lookForMouth = imcomplement(lookForMouth);
            %
            %                 SE = strel('disk',1);
            %                 lookForMouth = imdilate(lookForMouth, SE);
            %
            %                 regStats = regionprops(lookForMouth);
            %
            %                 cent = regStats.Centroid;
            %                 centRow = round(cent(2));
            %                 centCol = round(cent(1));
            %
            %
            %                 mouthBox = EmotionFace(centRow - 20: centRow + 5, centCol - 30: centCol + 30);
            %
            %
            %
            %             end
            
            
            
            mouthBox = EmotionFace(end-30: end, 50 - 20: 50 + 20);
            
            
            [featureVectorM,hogVisualizationM] = extractHOGFeatures(mouthBox,'CellSize',[4 4]);
            imshow(mouthBox)
            hold on;
            plot(hogVisualizationM);
%             
            
            
            %             subplot(1,2,1)
            %             imshow(foundEyesIm)
            %                         subplot(1,2,2)
            
            %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            HOGvector = horzcat(featureVectorE, featureVectorM);
            %                 %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            pause(0.005)
            
            
            
            
        end
        
        EmotionCheckIndex = strfind(Emotion, 'Crop');
        EmotionCheck = Emotion(1:EmotionCheckIndex-1);
        
        if (strcmp(EmotionCheck,'Angry'))
            label = 'A';
        elseif (strcmp(EmotionCheck,'Contempt'))
            label = 'C';
        elseif (strcmp(EmotionCheck,'Disgust'))
            label = 'D';
        elseif (strcmp(EmotionCheck,'Fear'))
            label = 'F';
        elseif (strcmp(EmotionCheck,'Happy'))
            label = 'H';
        elseif (strcmp(EmotionCheck,'Sad'))
            label = 'S';
        elseif (strcmp(EmotionCheck,'Surprise'))
            label = 'O';
        end
        
        FaceHOGs(end+1, :) = HOGvector;
        FaceLabels = [FaceLabels; label];
        
    end
    
end

FaceHOGs = FaceHOGs(2:end,:);
FaceLabels = FaceLabels(2:end);

MCSVM = fitcecoc(FaceHOGs,FaceLabels);
