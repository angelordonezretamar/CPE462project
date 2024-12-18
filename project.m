    % Angel Ordonez Retamar CPE462 Final Project
    % Completed working alone
    % I pledge my honor that I have abided by the Stevens Honor System
    % SUBBAND_IMAGE_CODER: Implements a wavelet-based image coder with quantization and entropy coding.

    % User Input for Quantization Step Size
    step_size = input('Enter the quanth\\\ization step size (e.g., 20): ');
    if isempty(step_size) || step_size <= 0
        error('Invalid step size. Please enter a positive value.');
    end

    % Read Input Image
    original_img = imread("guernica.jpg");
    if size(original_img, 3) > 1
        original_img = rgb2gray(original_img); % Convert to grayscale if necessary
    end
    original_img = double(original_img);

    % Wavelet Decomposition
    [cA, cH, cV, cD] = dwt2(original_img, 'haar'); % Haar wavelet decomposition

    % Store the size of cA and the original image for reconstruction later
    size_cA = size(cA);
    size_original = size(original_img);

    % Combine coefficients into a single structure for ease of processing
    subbands = struct('cA', cA, 'cH', cH, 'cV', cV, 'cD', cD);

    % Scalar Quantization
    quantized = structfun(@(coeff) round(coeff / step_size), subbands, 'UniformOutput', false);

    % Entropy Encoding
    quantized_data = [quantized.cA(:); quantized.cH(:); quantized.cV(:); quantized.cD(:)];
    symbols = unique(quantized_data);
    probabilities = histcounts(quantized_data, [symbols; max(symbols)+1]) / numel(quantized_data);
    huff_dict = huffmandict(symbols, probabilities);
    encoded_bits = huffmanenco(quantized_data, huff_dict);

    % Save encoded bitstream to a file
    save('encoded_stream.mat', 'encoded_bits', 'huff_dict', 'step_size', 'size_cA', 'size_original');

    fprintf('Encoding complete. Encoded bitstream saved as "encoded_stream.mat".\n');

    % Decoder - Entropy Decoding
    load('encoded_stream.mat', 'encoded_bits', 'huff_dict', 'step_size', 'size_cA', 'size_original');
    decoded_data = huffmandeco(encoded_bits, huff_dict);

    % Reconstruct quantized subbands
    num_cA = prod(size_cA); % Total elements in cA
    cA_decoded = reshape(decoded_data(1:num_cA), size_cA);
    offset = num_cA;
    cH_decoded = reshape(decoded_data(offset+1:offset+numel(quantized.cH)), size(quantized.cH));
    offset = offset + numel(quantized.cH);
    cV_decoded = reshape(decoded_data(offset+1:offset+numel(quantized.cV)), size(quantized.cV));
    offset = offset + numel(quantized.cV);
    cD_decoded = reshape(decoded_data(offset+1:end), size(quantized.cD));

    % Dequantize the subbands
    cA_rec = cA_decoded * step_size;
    cH_rec = cH_decoded * step_size;
    cV_rec = cV_decoded * step_size;
    cD_rec = cD_decoded * step_size;

    % Subband Reconstruction
    reconstructed_img = idwt2(cA_rec, cH_rec, cV_rec, cD_rec, 'haar');

    % Crop the reconstructed image to match the original dimensions
    reconstructed_img = reconstructed_img(1:size_original(1), 1:size_original(2));

    % PSNR
    mse = mean((original_img(:) - reconstructed_img(:)).^2);
    psnr = 10 * log10(255^2 / mse);

    % Display Results
    fprintf('PSNR of Reconstructed Image: %.2f dB\n', psnr);

    % Plot the images
    figure;
    subplot(1,2,1); imshow(uint8(original_img)); title('Original Image');
    subplot(1,2,2); imshow(uint8(reconstructed_img)); title('Reconstructed Image');