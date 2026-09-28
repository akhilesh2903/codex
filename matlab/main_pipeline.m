% main_pipeline.m
% ============================================================
% DR-Screening-XAI: MATLAB Image Processing Baseline Pipeline
% ============================================================
% Demonstrates MATLAB-based fundus image processing using:
%   - Image Processing Toolbox (CLAHE, morphology, filtering)
%   - Computer Vision Toolbox (feature extraction)
%   - Statistics and Machine Learning Toolbox (metrics)
%   - Medical Imaging Toolbox (where applicable)
%
% Usage (MATLAB command window):
%   results = main_pipeline('path/to/fundus.jpg');
%   main_pipeline('path/to/fundus.jpg', 'outputs/matlab_baseline');
%
% Or run as standalone script with default demo image:
%   run main_pipeline.m
% ============================================================

function results = main_pipeline(image_path, output_dir)
    % Default arguments
    if nargin < 1
        % Try to find a sample image
        candidates = {'data/sample.jpg', 'data/sample.png', ...
                      'data/raw/aptos/train_images'};
        image_path = '';
        for i = 1:length(candidates)
            if exist(candidates{i}, 'file') || exist(candidates{i}, 'dir')
                if exist(candidates{i}, 'dir')
                    d = dir(fullfile(candidates{i}, '*.png'));
                    if isempty(d), d = dir(fullfile(candidates{i}, '*.jpg')); end
                    if ~isempty(d)
                        image_path = fullfile(candidates{i}, d(1).name);
                    end
                else
                    image_path = candidates{i};
                end
                break;
            end
        end
        if isempty(image_path)
            error('No sample image found. Provide image_path as argument.');
        end
    end
    if nargin < 2
        output_dir = 'outputs/matlab_baseline';
    end

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    [~, fname, ~] = fileparts(image_path);
    fprintf('\n════════════════════════════════════════════\n');
    fprintf('  DR-Screening-XAI MATLAB Pipeline\n');
    fprintf('════════════════════════════════════════════\n');
    fprintf('Image: %s\n\n', image_path);

    % ── Step 1: Load Image ──────────────────────────────────────────────────
    fprintf('[1/8] Loading image...\n');
    img = imread(image_path);
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);
    end
    [H, W, ~] = size(img);
    fprintf('      Dimensions: %d x %d\n', W, H);

    % ── Step 2: Image Quality Assessment ───────────────────────────────────
    fprintf('[2/8] Image quality assessment...\n');
    gray = rgb2gray(img);

    % Blur — Variance of Laplacian
    lap_filter = fspecial('laplacian', 0.5);
    img_lap = imfilter(double(gray), lap_filter, 'replicate');
    blur_score = var(img_lap(:));

    % Brightness and contrast
    brightness = mean(double(gray(:)));
    contrast_score = std(double(gray(:)));

    % Retinal area (fraction of non-black pixels)
    retina_mask = gray > 10;
    field_of_view = sum(retina_mask(:)) / numel(retina_mask);

    % Quality score (weighted)
    norm_blur = min(blur_score / 1000.0, 1.0);
    norm_brightness = 1.0 - abs(brightness - 127.5) / 127.5;
    norm_contrast = min(contrast_score / 100.0, 1.0);
    quality_score = norm_blur * 0.4 + norm_brightness * 0.2 + norm_contrast * 0.2 + field_of_view * 0.2;

    if quality_score >= 0.7
        quality_status = 'GOOD';
    elseif quality_score >= 0.4
        quality_status = 'BORDERLINE';
    else
        quality_status = 'UNGRADABLE';
    end

    fprintf('      Quality Score: %.3f (%s)\n', quality_score, quality_status);
    fprintf('      Blur Score: %.2f | Brightness: %.1f | Contrast: %.1f\n', ...
            blur_score, brightness, contrast_score);

    % ── Step 3: Illumination Normalization ──────────────────────────────────
    fprintf('[3/8] Illumination normalization (morphological background subtraction)...\n');
    hsv = rgb2hsv(im2double(img));
    v = hsv(:, :, 3);
    se = strel('disk', 35);
    bg = imopen(v, se);
    v_norm = v - bg;
    v_norm = imadjust(v_norm);
    hsv(:, :, 3) = v_norm;
    img_norm = im2uint8(hsv2rgb(hsv));

    % ── Step 4: CLAHE Enhancement ───────────────────────────────────────────
    fprintf('[4/8] CLAHE contrast enhancement...\n');
    lab = rgb2lab(im2double(img_norm));
    L = lab(:, :, 1) / 100;
    L_clahe = adapthisteq(L, 'NumTiles', [8 8], 'ClipLimit', 0.02);
    lab(:, :, 1) = L_clahe * 100;
    img_enhanced = im2uint8(lab2rgb(lab));

    % ── Step 5: Blood Vessel Segmentation (Green channel) ───────────────────
    fprintf('[5/8] Blood vessel segmentation (green channel morphology)...\n');
    green = img_enhanced(:, :, 2);
    green_d = im2double(green);

    % CLAHE on green channel
    green_clahe = adapthisteq(green_d, 'NumTiles', [8 8], 'ClipLimit', 0.03);

    % Vessel enhancement via morphological bottom-hat (dark structures on bright BG)
    se_vessel = strel('disk', 5);
    tophat = imtophat(green_clahe, se_vessel);

    % Adaptive threshold
    thresh = adaptthresh(tophat, 0.4, 'ForegroundPolarity', 'bright');
    vessels_bw = ~imbinarize(tophat, thresh);  % invert: vessels are dark

    % Remove small noise (<100px)
    vessels_clean = bwareaopen(vessels_bw, 100);

    % ── Step 6: Microaneurysm Candidate Detection ───────────────────────────
    fprintf('[6/8] Microaneurysm candidate detection (tophat morphology)...\n');
    se_ma = strel('disk', 7);
    blackhat = imbothat(green_clahe, se_ma);
    thresh_ma = adaptthresh(blackhat, 0.5);
    ma_bw = imbinarize(blackhat, thresh_ma);
    ma_bw = bwareaopen(ma_bw, 5);
    ma_bw = ~bwareaopen(~ma_bw, 200);  % remove large regions
    ma_stats = regionprops(ma_bw, 'Area', 'Centroid', 'BoundingBox');
    ma_count = length(ma_stats);
    fprintf('      Microaneurysm candidates: %d\n', ma_count);

    % ── Step 7: Optic Disc Localization (brightest region) ──────────────────
    fprintf('[7/8] Optic disc localization...\n');
    green_gray = rgb2gray(img_enhanced);
    se_disc = strel('disk', 30);
    smoothed = imopen(green_gray, se_disc);
    [~, max_idx] = max(smoothed(:));
    [od_y, od_x] = ind2sub(size(smoothed), max_idx);
    fprintf('      Optic disc centroid: (%d, %d)\n', od_x, od_y);

    % ── Step 8: Save Outputs & Generate Figures ─────────────────────────────
    fprintf('[8/8] Saving outputs and figures...\n');

    % Save processed images
    imwrite(img_enhanced, fullfile(output_dir, sprintf('%s_enhanced.png', fname)));
    imwrite(vessels_clean, fullfile(output_dir, sprintf('%s_vessels.png', fname)));
    imwrite(ma_bw, fullfile(output_dir, sprintf('%s_ma_candidates.png', fname)));

    % Generate comparison figure
    fig1 = figure('Visible', 'off', 'Position', [0, 0, 1400, 500]);
    subplot(1, 4, 1); imshow(img); title('Original Fundus', 'FontSize', 11);
    subplot(1, 4, 2); imshow(img_enhanced); title('CLAHE Enhanced', 'FontSize', 11);
    subplot(1, 4, 3); imshow(vessels_clean); title('Vessel Segmentation', 'FontSize', 11);
    subplot(1, 4, 4); imshow(img_enhanced); hold on;
    viscircles([od_x, od_y], 40, 'Color', 'yellow', 'LineWidth', 2);
    if ~isempty(ma_stats)
        for i = 1:min(length(ma_stats), 50)
            c = ma_stats(i).Centroid;
            plot(c(1), c(2), 'r.', 'MarkerSize', 8);
        end
    end
    title('Lesion Candidates + OD', 'FontSize', 11); hold off;
    sgtitle(sprintf('MATLAB DR Processing Pipeline — %s', fname), 'FontSize', 13, 'FontWeight', 'bold');
    saveas(fig1, fullfile(output_dir, sprintf('%s_pipeline_comparison.png', fname)));
    close(fig1);

    % Quality metrics figure
    fig2 = figure('Visible', 'off', 'Position', [0, 0, 400, 300]);
    bar([quality_score, norm_blur, norm_brightness, norm_contrast, field_of_view], ...
        'FaceColor', [0.2 0.4 0.8]);
    set(gca, 'XTickLabel', {'Quality', 'Blur', 'Brightness', 'Contrast', 'FOV'});
    ylabel('Score (0–1)'); title('Image Quality Metrics'); ylim([0, 1.1]);
    saveas(fig2, fullfile(output_dir, sprintf('%s_quality_metrics.png', fname)));
    close(fig2);

    % ── Return results struct ────────────────────────────────────────────────
    results.image_path = image_path;
    results.quality_score = quality_score;
    results.quality_status = quality_status;
    results.blur_score = blur_score;
    results.brightness = brightness;
    results.contrast_score = contrast_score;
    results.field_of_view = field_of_view;
    results.ma_candidates = ma_count;
    results.optic_disc = [od_x, od_y];
    results.output_dir = output_dir;

    fprintf('\n════════════════════════════════ DONE ═════\n');
    fprintf('Outputs saved to: %s\n', output_dir);
    fprintf('  %s_enhanced.png\n', fname);
    fprintf('  %s_vessels.png\n', fname);
    fprintf('  %s_ma_candidates.png\n', fname);
    fprintf('  %s_pipeline_comparison.png\n', fname);
    fprintf('  %s_quality_metrics.png\n', fname);

end
