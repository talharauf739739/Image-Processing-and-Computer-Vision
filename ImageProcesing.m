    function Iout = ImageProcesing(newImage)
                
        I = imread(newImage);
        if ismatrix(I)
            I = cat(3,I,I,I);
        end

        % Resize the image as required for the CNN.
        Iout = imresize(I, [299 299]);

        % Note that the aspect ratio is not preserved. In Caltech 101, the
        % object of interest is centered in the image and occupies a
        % majority of the image scene. Therefore, preserving the aspect
        % ratio is not critical. However, for other data sets, it may prove
        % beneficial to preserve the aspect ratio of the original image
        % when resizing.
    end