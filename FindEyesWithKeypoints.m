function [ eyeBalls ] = FindEyesWithKeypoints( CropFace )

%Choose a keypoint detection method
corners = detectHarrisFeatures(CropFace);
% corners = detectFASTFeatures(CropFace);

%Select number of strongest keypoints to keep, and the locations of the
%keypoints
strongestCorners = corners.selectStrongest(5);
strongestCornerLocations = strongestCorners.Location;

boxNumber = 1;
maxCount = 0;

for i = 1:10
    
    topRangeRow = (i-1)*10 +1;
    bottomRangeRow = i*10; 
    
    keypointCount = 0;
    
    for j = 1:length(strongestCornerLocations(:,1))
       
        rowLocation = strongestCornerLocations(j,2);
        
        %check if keypoints are within the current interval
        if rowLocation >= topRangeRow && rowLocation <= bottomRangeRow
            keypointCount = keypointCount + 1;
        end
        
    end
    
    %if new max found in current interval, set max to current count
    %and note box number
    if keypointCount > maxCount
        maxCount = keypointCount;
        boxNumber = i;
    end 
end

eyeBalls = CropFace( (boxNumber-1)*10+1 - 20 : boxNumber*10 + 10, 15:85);

end

