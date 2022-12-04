import os
import struct

import pyaudio
import wave
import soundfile as sf

import numpy as np
from data_preprocessing.utils import read_audio
from data_preprocessing.feature_extractor import FeatureExtractor
import IPython
from keras.models import load_model
import IPython.display as ipd


# from recording_helper import record_audio,terminate

windowLength = 256
overlap = round(0.25 * windowLength)  # overlap of 75%
ffTLength = windowLength
inputFs = 16e3
fs = 16e3
sr=16000
numFeatures = ffTLength // 2 + 1
numSegments = 8


def prepare_input_features(stft_features):
    noisySTFT = np.concatenate([stft_features[:,0:numSegments-1], stft_features], axis=1)
    stftSegments = np.zeros((numFeatures, numSegments , noisySTFT.shape[1] - numSegments + 1))

    for index in range(noisySTFT.shape[1] - numSegments + 1):
        stftSegments[:,:,index] = noisySTFT[:,index:index + numSegments]
    return stftSegments

def get_predictor(noisyAudio):
    noiseAudioFeatureExtractor = FeatureExtractor(noisyAudio, windowLength=windowLength, overlap=overlap,
                                                  sample_rate=sr)
    noise_stft_features = noiseAudioFeatureExtractor.get_stft_spectrogram()

    noisyPhase = np.angle(noise_stft_features)
    noise_stft_features = np.abs(noise_stft_features)

    mean = np.mean(noise_stft_features)
    std = np.std(noise_stft_features)
    noise_stft_features = (noise_stft_features - mean) / std
    predictors = prepare_input_features(noise_stft_features)
    predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1, predictors.shape[2]))
    predictors = np.transpose(predictors, (3, 0, 1, 2)).astype(np.float32)
    return predictors, noiseAudioFeatureExtractor, noisyPhase, mean, std


def load_trained_model():
    model = load_model('C:\\Users\\B989\\PycharmProjects\\SPL3_FinalAPP\\prediction\\denoiser_cnn_log_mel_generator_v2.h5')
    return model


def predict(model, predictors):
    STFTFullyConvolutional = model.predict(predictors)
    return STFTFullyConvolutional


def revert_features_to_audio(noiseAudioFeatureExtractor, features, phase, cleanMean=None, cleanStd=None):
    # scale the outpus back to the original range
    if cleanMean and cleanStd:
        features = cleanStd * features + cleanMean

    phase = np.transpose(phase, (1, 0))
    features = np.squeeze(features)

    features = features * np.exp(1j * phase)  # that fixes the abs() ope previously done
    #
    features = np.transpose(features, (1, 0))
    return noiseAudioFeatureExtractor.get_audio_from_stft_spectrogram(features)


def process_audio(audio):
    predictors, noiseAudioFeatureExtractor, noisyPhase, mean, std = get_predictor(audio)
    model = load_trained_model()
    STFTFullyConvolutional = predict(model, predictors)
    denoisedAudioFullyConvolutional = revert_features_to_audio(noiseAudioFeatureExtractor, STFTFullyConvolutional,
                                                               noisyPhase, mean, std)
    return denoisedAudioFullyConvolutional


fs = 16e3
filename = "recorded_denoise1.wav"


def denoise_from_wav_file(file):
    print(file)
    audio, sr = read_audio(file,
        sample_rate=fs)

    print(type(audio))
    audio = process_audio(audio)

    sf.write(filename, audio, 16000)
    return os.path.join(os.getcwd(), filename)

    # print(audio)
    # wav_file = wave.open(filename, 'wb')
    # wav_file.setnchannels(1)
    # wav_file.setsampwidth(2)
    # wav_file.setframerate(16000)
    # wav_file.writeframes(b''.join(audio))
    # wav_file.close()

    return os.path.join(os.getcwd(), filename)



if __name__ == '__main__':
    pass

# def preprocess_audiobuffer(waveform):
#     """
#     waveform: ndarray of size (16000, )
#
#     output: Spectogram Tensor of size: (1, `height`, `width`, `channels`)
#     """
#     #  normalize from [-32768, 32767] to [-1, 1]
#     waveform = waveform / 32768
#
#     waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
#
#     spectogram = get_spectrogram(waveform)
#
#     # add one dimension
#     spectogram = tf.expand_dims(spectogram, 0)
#
#     return spectogram


# if __name__ == "__main__":
#     filename = "recorded.wav"
#     frames = []
#     for i in range(20):
#         audio = record_audio()
#         print(audio)
#         # waveform = audio / 32768
#         denoised_audio = process_audio(audio)
#         frame = denoised_audio.tobytes()
#         frames.append(frame)
#
#     # data = struct.pack('<' + ('h'*len(signal)), *signal)
#     wav_file = wave.open(filename, 'wb')
#     wav_file.setnchannels(1)
#     wav_file.setsampwidth(2)
#     wav_file.setframerate(16000)
#     wav_file.writeframes(b''.join(frames))
#     wav_file.close()

