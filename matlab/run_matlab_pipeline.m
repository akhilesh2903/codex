% run_matlab_pipeline.m
% DR-Screening-XAI Image Processing Pipeline Baseline

function run_matlab_pipeline(image_path, output_dir)
    if nargin < 2
        output_dir = 'outputs/matlab_baseline';
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % 1. Load image
    fprintf('Processing %s\n', image_path);
    img = imread(image_path);
    
    % 2. Quality Assessment
    % Blur using variance of laplacian
    gray = rgb2gray(img);
    lap = fspecial('laplacian');
    img_lap = imfilter(double(gray), lap, 'replicate');
    blur_score = var(img_lap(:));
    fprintf('Blur Score (Variance of Laplacian): %.2f\n', blur_score);
    
    % 3. Illumination Normalization
    % Background subtraction using morphological opening
    hsv = rgb2hsv(img);
    v = hsv(:,:,3);
    se = strel('disk', 35);
    bg = imopen(v, se);
    v_norm = v - bg;
    v_norm = imadjust(v_norm);
    hsv(:,:,3) = v_norm;
    img_norm = hsv2rgb(hsv);
    
    % 4. CLAHE Enhancement
    lab = rgb2lab(img_norm);
    L = lab(:,:,1)/100;
    L_clahe = adapthisteq(L, 'NumTiles', [8 8], 'ClipLimit', 0.02);
    lab(:,:,1) = L_clahe * 100;
    img_enhanced = lab2rgb(lab);
    
    % 5. Save Output
    [~, name, ext] = fileparts(image_path);
    out_file = fullfile(output_dir, sprintf('%s_enhanced%s', name, ext));
    imwrite(img_enhanced, out_file);
    fprintf('Saved enhanced image to %s\n', out_file);
    
    % 6. Basic Vessel Segmentation (Experimental)
    green = img_enhanced(:,:,2);
    vessels = adaptthresh(green, 0.4);
    vessels_bw = imbinarize(green, vessels);
    vessels_clean = bwareaopen(~vessels_bw, 100);
    out_vessel_file = fullfile(output_dir, sprintf('%s_vessels%s', name, ext));
    imwrite(vessels_clean, out_vessel_file);
    fprintf('Saved vessel mask to %s\n', out_vessel_file);

end
