
%Reduce dimensionality of all emotions to allow for Fisher LDA

EmotionsForFisherface = {'Angry', 'Fear', 'Happy', 'Sad', 'Surprise'};

EmotionFisherFaces = zeros(5,10000);

for k = 1:length(EmotionsForFisherface)
    
    EmotionTrain = EmotionsForFisherface{k};
    
    CropSize = 100;
    
    numEigFaces = 100;
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    %REDUCE DIMENSIONALITY OF HAPPY VS NONHAPPY FACES
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    %Reduce cropped Happy Images by projecting onto top 100 Eigenfaces
    
    HappyReduce = zeros(numEigFaces,1);
    %read in all cropped face images and subtract total mean face
    CropName = strcat(EmotionTrain,'Crop');
    HappyCrop = dir(fullfile(CropName, '*.png'));
    
    HappyFaces = zeros(1, CropSize, CropSize);
    
    for i = 1:length(HappyCrop)
        
        Im = im2double(imread(fullfile(CropName, HappyCrop(i).name)));  % Read image
        
        HappyFaces(end+1,:,:) = Im;
        
        Im = reshape(Im, [CropSize^2 1]); %reshape image into vector
        v = zeros(1,1);
        
        for j = 1:numEigFaces
            ProjC = dot(Im, top100Eigenfaces(:,j));
            v(end+1,1) = ProjC;
        end
        
        v = v(2:end,1);
        HappyReduce(:, end+1) = v;
        
        
    end
    
    HappyReduce = HappyReduce(:,2:end);
    HappyFaces = HappyFaces(2:end,:,:);
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    %Reduce cropped Non-Happy Images by projecting onto top 100 Eigenfaces
    
    NonHappyReduce = zeros(numEigFaces,1);
    
    TotalCrop = dir(fullfile('TotalCrop', '*.png'));
    
    NonHappyFaces = zeros(1,CropSize,CropSize);
    
    for i = 1:length(TotalCrop)
        
        Im = im2double(imread(fullfile('TotalCrop', TotalCrop(i).name)));  % Read image
        filename = TotalCrop(i).name; %get the current file's name
        strIndex = strfind(filename,EmotionTrain); %check if 'Happy' is in the filename
        
        %if filename does not contain 'Happy', reduce dimensionality via
        %projection on top 100 eigenfaces and add coefficient vector to
        %Non-happy group
        if isempty(strIndex)
            
            NonHappyFaces(end+1,:,:) = Im(:,:);
            
            Im = reshape(Im, [CropSize^2 1]); %reshape image into vector
            v = zeros(1,1);
            
            for j = 1:numEigFaces
                ProjC = dot(Im, top100Eigenfaces(:,j));
                v(end+1,1) = ProjC;
            end
            
            v = v(2:end,1);
            NonHappyReduce(:, end+1) = v;
            
        end
        
    end
    
    NonHappyFaces = NonHappyFaces(2:end,:,:);
    NonHappyReduce = NonHappyReduce(:,2:end);
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    %Calculate the mean faces for the Happy and Non-Happy sets, as well as
    %total mean face
    TotalReduceSum = HappyReduce(:,1);
    
    HappyReduceSum = HappyReduce(:,1);
    for i = 2:length(HappyReduce(1,:))
        HappyReduceSum = HappyReduceSum + HappyReduce(:,i);
        TotalReduceSum = TotalReduceSum + HappyReduce(:,i);
    end
    HappyMeanReduce = HappyReduceSum/length(HappyReduce(1,:));
    
    
    NonHappyReduceSum = NonHappyReduce(:,1);
    for i = 2:length(NonHappyReduce(1,:))
        NonHappyReduceSum = NonHappyReduceSum + NonHappyReduce(:,i);
        TotalReduceSum = TotalReduceSum + NonHappyReduce(:,i);
    end
    NonHappyMeanReduce = NonHappyReduceSum/length(NonHappyReduce(1,:));
    
    TotalMeanReduce = TotalReduceSum/241;
    
    %set up sub Rb matrix, with happy mean and nonhappy mean faces concatenated
    %then subtract total mean face from both
    Rb_sub = HappyMeanReduce;
    Rb_sub(:,end+1) = NonHappyMeanReduce;
    Rb_sub(:,1) = Rb_sub(:,1) - TotalMeanReduce;
    Rb_sub(:,2) = Rb_sub(:,2) - TotalMeanReduce;
    
    
    % set up sub Rw matrix
    RwH = HappyReduce;
    RwNH = NonHappyReduce;
    
    for i = 1:length(RwH(1,:))
        RwH(:,i) = RwH(:,i) - HappyMeanReduce;
    end
    
    for i = 1:length(RwNH(1,:))
        RwNH(:,i) = RwNH(:,i) - NonHappyMeanReduce;
    end
    
    Rw_sub = RwH;
    for i = 1:length(RwNH(1,:))
        Rw_sub(:, end+1) = RwNH(:,i);
    end
    
    Rb = Rb_sub*Rb_sub';
    Rw = Rw_sub*Rw_sub';
    
    [eVecs, eVals] = eig(inv(Rw)*Rb);
    [M,I] = max(eVals(:));
    
    eVecs = (inv(Rw)*Rb)*eVecs;
    
    % eVecs = fliplr(eVecs);
    
    fishvec = eVecs(:,1);
    
    fisherface = eVecs(1)*top100Eigenfaces(:,1);
    for i = 2:numEigFaces
        fisherface = fisherface+eVecs(i)*top100Eigenfaces(:,i);
    end
    fisherface = fisherface/norm(fisherface);
    
    EmotionFisherFaces(k,:) = fisherface;
    
    figure
    
    imshow(squeeze(reshape(fisherface, CropSize, CropSize, 1)),[])
    title(strcat('Fisherface to distinguish:', '"', EmotionTrain, '"'))
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    vecProjF_Happy = zeros(1);
    vecProjF_NonHappy = zeros(1);
    
    TotalMeanFace = reshape(TotalMeanFace, [CropSize^2 1]);
    
    for i = 1:length(HappyFaces(:,1,1))
        
        ImH = HappyFaces(i,:,:);
        ImH = squeeze(ImH);
        
        ImH = reshape(ImH, [CropSize^2 1]);
        
        ImH = ImH - TotalMeanFace;
        
        
        w = dot(fisherface,ImH);
        
        vecProjF_Happy(end+1) = w;
    end
    vecProjF_Happy = vecProjF_Happy(:,2:end);
    
    for i = 1:length(NonHappyFaces(:,1,1))
        
        ImNH = NonHappyFaces(i,:,:);
        ImNH = squeeze(ImNH);
        
        ImNH = reshape(ImNH, [CropSize^2 1]);
        
        ImNH = ImNH - TotalMeanFace;
        
        wf = dot(fisherface,ImNH);
        
        vecProjF_NonHappy(end+1) = wf;
        
    end
    vecProjF_NonHappy = vecProjF_NonHappy(:,2:end);
    
    vecProjF_Happy = sort(vecProjF_Happy);
    vecProjF_NonHappy = sort(vecProjF_NonHappy);
    
        figure
        hist(vecProjF_Happy,30)
        hf = findobj(gca,'Type','patch');
        set(hf,'FaceColor','r','EdgeColor','w','facealpha',0.75)
        hold on;
        hist(vecProjF_NonHappy,30)
        hf2 = findobj(gca,'Type','patch');
        set(hf2,'facealpha',0.75);
        axis([-5 5 0 20])
        title('Training data projected on Fisherface');
        % line([-0.5 -0.5], [0 20]);
        legend(EmotionTrain, strcat('Non-',EmotionTrain));
    
        TestFaces = dir(fullfile('TestFacesCrop', '*.png'));
        
    for i = 1:length(TestFaces)
        
        Im = im2double(imread(fullfile('TestFacesCrop', TestFaces(i).name)));  % Read image
        filename = TestFaces(i).name; %get the current file's name
        
        Im = reshape(Im, [CropSize^2 1]);
        
        coeff = dot(fisherface,Im);
        
        
%         if coeff > -1
%             fprintf (strcat(filename, ' -  Surprise\n'))
%         else
%             fprintf(strcat(filename, ' -  Not Surprise\n'))
%         end
    end
    
end