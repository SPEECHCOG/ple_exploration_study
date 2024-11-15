function final = filter_files(source_dir, target_dir, masking)

if nargin < 3
    masking = 0;
end

% source_dir  Define the parent directory path where the audio files are located


% target_dir  Define the output directory path where the filtered audio files will be saved


% Create output_dir if it doesn't exist
if ~exist(target_dir, 'dir')
    mkdir(target_dir);
end

% Define the filter coefficients according to (Gerhardt and Abrams, 1996)
% model of fetal isolation (developing hearing system).
% Attenuatio of 10 dB for 100 Hz; 35 dB for 500 Hz; and 44 dB for 2 kHz
a = [ 1.0000   -1.9560    0.9569];
b = 1.0e-03 * [0.2373    0.4746    0.2373];

% Recursively search for all audio files in the parent directory
audio_files = dir(fullfile(source_dir, '**/*.wav'));

% Loop through each audio file and apply the filter
for i = 1:length(audio_files)
    % Get the file path and name
    file_path = fullfile(audio_files(i).folder, audio_files(i).name);
    
    % Load the audio data
    [signal, fs] = audioread(file_path);
    
    
    % Noise simulation (simulate masking)
    if(masking)
        target_snr = 10;
        noise = randn(size(signal)).*0.01;

        X = sqrt(mean(signal.^2));
        S = sqrt(mean(noise.^2));

        % Energy of the noise

        noise_scaler = 10^(-target_snr/20);

        noise = noise.*X/S.*noise_scaler;

        % Ensure SNR is ok
        SNR = 10*log10((signal'*signal)/(noise'*noise));
        signal = signal + noise;
    end
    
    
    % Apply the filter
    filtered_signal = filter(b, a, signal);
    % normalised amplitud
    filtered_signal = filtered_signal ./max(abs(filtered_signal));
    
    % Define the output file path and name
    full_path_source = dir(source_dir).folder;
    full_path_target = dir(target_dir).folder;
    output_file_path = strrep(file_path, full_path_source, full_path_target);
    
    % Make sure the output directory exists
    [output_dir_path, ~, ~] = fileparts(output_file_path);
    if ~exist(output_dir_path, 'dir')
        mkdir(output_dir_path);
    end
    
    % Save the filtered audio data
    audiowrite(output_file_path, filtered_signal, fs);
end

final = 'success';
