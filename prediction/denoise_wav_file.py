# import os
import os
import wave

from data_preprocessing.utils import read_audio
from live_audio_testing import process_audio
import soundfile as sf

fs = 16e3
filename = "recorded_denoise.wav"


def denoise_from_wav_file(file):
    audio, sr = read_audio(file,
        sample_rate=fs)

    audio = process_audio(audio)
    # wav_file = wave.open(filename, 'wb')
    # wav_file.setnchannels(1)
    # wav_file.setsampwidth(2)
    # wav_file.setframerate(16000)
    # wav_file.writeframes(b''.join(audio))
    # wav_file.close()

    sf.write(filename, audio, 16000)
    return os.path.join(os.getcwd(), filename)



if __name__ == '__main__':
    f = denoise_from_wav_file('C:/Users/B989/OneDrive - Brain Station 23 Ltd/Documents/crblp_speech_corpus_release_V1.0/crblp_speech_corpus_release_V1.0/wav/21234.wav')
    print(f)