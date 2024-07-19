import pyaudio
import wave
import time
import os
import struct
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa
import sounddevice as sd

from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter, lfilter

# Define parameters
chunk = 2048  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 32 bits per sample
data_type = np.int16
channels = 1
fs = 44100  # Record at 44100 samples per second
max_duration = 2000 # Maximum duration of each audio file in seconds

time_interval = 1 # Time interval for recording
transcribe_interval = 20 # Time interval for transcribing
speed_factor = 3.0 # Speed up audio
output_dir = 'recordings'  # Directory to save recordings
output_dir_fast = 'recordings_fast'  # Directory to save recordings
sound_increase = 10  # Increase volume by X dB
DEVICE_INDEX = 3          # Replace with the index of your microphone

# Input Device id  0  -  Microsoft Sound Mapper - Input
# Input Device id  1  -  Microphone (NVIDIA Broadcast)
# Input Device id  2  -  Microphone (RODE NT-USB)
# Input Device id  3  -  CABLE Output (VB-Audio Virtual 
# Input Device id  4  -  Microphone (HD Pro Webcam C920)


# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_fast, exist_ok=True)

# Remove existing files in the output directory
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

for file in os.listdir(output_dir_fast):
    os.remove(os.path.join(output_dir_fast, file))

# Initialize PyAudio object
p = pyaudio.PyAudio()

# Function to record audio in chunks and save to file
def record_and_save():
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input_device_index=DEVICE_INDEX,
                    input=True)

    frames = []
    counter = 1
    wav_filename = f"{output_dir}/audio_{counter}.wav"

    start_time = time.time()
    start_time_global = time.time()
    current_duration = 0

    fig, ax = plt.subplots()
    x = np.arange(0, 2 * chunk, 2)
    line, = ax.plot(x, np.random.rand(chunk))
    # plt.ylim(-2**31, 2**31)  # Range for paInt32
    # plt.ylim(-2**15, 2**15)  # Range for paInt16
    plt.ylim(-32768, 32767)  # Range for 16-bit PCM
    plt.xlim(0, chunk)
    plt.show(block=False)

    while True:
        try:
            data = stream.read(chunk)
            frames.append(data)

            # Update plot
            audio_data = np.frombuffer(data, dtype=data_type)
            line.set_ydata(audio_data)
            fig.canvas.draw()
            fig.canvas.flush_events()

        except IOError as e:
            print(f"Error recording: {e}")
            continue

        current_duration = time.time() - start_time
        
        if current_duration >= time_interval:
            int_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            if np.max(np.abs(int_data)) > 256:  # Detect significant audio
                if os.path.exists(wav_filename):
                    # Append to existing file
                    with wave.open(wav_filename, 'rb') as wf:
                        params = wf.getparams()
                        existing_frames = wf.readframes(params.nframes)
                    
                    with wave.open(wav_filename, 'wb') as wf:
                        wf.setparams(params)
                        wf.writeframes(existing_frames)
                        wf.writeframes(b''.join(frames))
                else:
                    # Create new file
                    with wave.open(wav_filename, 'wb') as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(p.get_sample_size(sample_format))
                        wf.setframerate(fs)
                        wf.writeframes(b''.join(frames))

                 # Increase volume and save the audio
                try:
                    if(os.path.exists(wav_filename)):
                        sound = AudioSegment.from_wav(wav_filename)
                        if(len(sound) > transcribe_interval*1000):
                            louder_sound = sound[-transcribe_interval*1000:] + sound_increase  # Increase volume by 10 dB
                        else:
                            louder_sound = sound + sound_increase
                        louder_sound.export(f"{output_dir_fast}/audio_{counter}.wav", format="wav")
                except Exception as e:
                    print(f"Error increasing volume: {e}")
                print(f"Saved {wav_filename}")
            else:
                start_time_global += time_interval # Increase time delay
                print("No Sound. Skip.")

           

            frames = []
            start_time = time.time()
            current_duration = 0

        # Check if the file duration has reached the max duration
        if (time.time() - start_time_global) >= max_duration:
            # counter += 1
            start_time_global = time.time()

            try:
                if(os.path.exists(wav_filename)):
                    sound = AudioSegment.from_wav(wav_filename)
                    if(len(sound) > transcribe_interval*1000):
                        louder_sound = sound[-transcribe_interval*1000:] + sound_increase  # Increase volume by 10 dB
                    else:
                        louder_sound = sound + sound_increase
                    louder_sound.export(f"{output_dir_fast}/audio_{counter}.wav", format="wav")
            except Exception as e:
                print(f"Error increasing volume: {e}")

            # os.remove(os.path.join(output_dir, file))
            # break

try:
    record_and_save()
except KeyboardInterrupt:
    print("Recording stopped")
finally:
    # Terminate the PyAudio object
    p.terminate()

# Close the chart
plt.close()

# Function to play the saved audio file
def play_audio_sa(filename):
    # Load the audio file
    audio = AudioSegment.from_wav(filename)
    # Convert to raw audio data
    play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
    print(f"Playing {play_obj}")
    # Wait for playback to finish
    play_obj.wait_done()

def play_audio(filename, device_id):
    # Load the audio file
    audio = AudioSegment.from_wav(filename)
    # Convert to raw audio data
    audio_data = np.array(audio.get_array_of_samples())
    # Set the playback device
    sd.default.device = device_id
    # Play the audio data
    sd.play(audio_data, samplerate=audio.frame_rate)
    print(f"Playing audio on device {device_id}")
    # Wait for playback to finish
    sd.wait()

# Play the saved audio file
wav_filename = f"{output_dir}/audio_1.wav"

# # Increase volume and save the audio
# sound = AudioSegment.from_wav(wav_filename)
# louder_sound = sound + sound_increase  # Increase volume by 10 dB
# louder_sound.export(wav_filename, format="wav")

def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Apply noise reduction # Not really helpful
# rate, data = wavfile.read(wav_filename)
# filtered_data = bandstop_filter(data, 60, 120, fs, order=2)  # Example for 60Hz noise
# wavfile.write(wav_filename, rate, np.int16(filtered_data))

# play_audio(wav_filename, 6)


# Process and save with increased speed
# sound = AudioSegment.from_wav(wav_filename)
# fast_sound = sound.speedup(playback_speed=2.0)
# fast_sound.export(f"{output_dir_fast}/audio_{counter}_fast.wav", format="wav")