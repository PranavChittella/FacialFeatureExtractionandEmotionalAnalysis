
%Perform PCA on all images to reduce dimensionality

%Predetermined cropped face sizes
CropSize = 100;
VecLength = CropSize^2;

%Matrix to hold all vector-reshaped face images
FaceVecs = zeros(VecLength, 1);

%Get the calculated total mean face
TotalMeanFace = im2double(imread('MeanFaces/TotalMeanFace.png'));

%read in all cropped face images and subtract total mean face 
TotalCrop = dir(fullfile('TotalCrop', '*.png'));

for j=1:length(TotalCrop)
    Im = im2double(imread(fullfile('TotalCrop', TotalCrop(j).name)));  % Read image
    
    %subtract total mean face from each image
    ImSub = Im - TotalMeanFace;
    
    ImSub = reshape(ImSub,[VecLength, 1]);
    
    FaceVecs(:, end+1) = ImSub;
    
end

FaceVecs = FaceVecs(:,2:end);

SHS = FaceVecs'*FaceVecs;
[eigVecs eigVals] = eig(SHS);

Sv = FaceVecs*eigVecs;
Sv = fliplr(Sv);

top100Eigenfaces = Sv(:, 1:200);

for i =  1:100
    top100Eigenfaces(:,i) = top100Eigenfaces(:,i)/norm(top100Eigenfaces(:,i));
end

eigImage1 = reshape(top100Eigenfaces(:,1), CropSize, CropSize);
% eigImage1 = mat2gray(eigImage1);
eigImage2 = reshape(top100Eigenfaces(:,2), CropSize, CropSize);
% eigImage2 = mat2gray(eigImage2);
eigImage3 = reshape(top100Eigenfaces(:,3), CropSize, CropSize);
% eigImage3 = mat2gray(eigImage3);
eigImage4 = reshape(top100Eigenfaces(:,4), CropSize, CropSize);
% eigImage4 = mat2gray(eigImage4);
eigImage5 = reshape(top100Eigenfaces(:,5), CropSize, CropSize);
% eigImage5 = mat2gray(eigImage5);
eigImage6 = reshape(top100Eigenfaces(:,6), CropSize, CropSize);
% eigImage6 = mat2gray(eigImage6);
eigImage7 = reshape(top100Eigenfaces(:,7), CropSize, CropSize);
% eigImage7 = mat2gray(eigImage7);
eigImage8 = reshape(top100Eigenfaces(:,8), CropSize, CropSize);
% eigImage8 = mat2gray(eigImage8);
eigImage9 = reshape(top100Eigenfaces(:,9), CropSize, CropSize);
% eigImage9 = mat2gray(eigImage9);
eigImage10 = reshape(top100Eigenfaces(:,10), CropSize, CropSize);
% eigImage10 = mat2gray(eigImage10);

figure
subplot(2,5,1)
imshow(eigImage1, [])
subplot(2,5,2),
imshow(eigImage2, [])
subplot(2,5,3),
imshow(eigImage3, [])
title('Top 10 Eigenfaces for All Face Images');
subplot(2,5,4),
imshow(eigImage4, [])
subplot(2,5,5),
imshow((eigImage5), [])
subplot(2,5,6),
imshow((eigImage6), [])
subplot(2,5,7),
imshow((eigImage7), [])
subplot(2,5,8),
imshow((eigImage8), [])
subplot(2,5,9),
imshow((eigImage9), [])
subplot(2,5,10),
imshow(eigImage10, [])


