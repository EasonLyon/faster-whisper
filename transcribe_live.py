from faster_whisper import WhisperModel, BatchedInferencePipeline
import time

model_size = "large-v3"
# model_size = "distil-large-v3" # English model (faster)
# model_size = "tiny"
# model_size = "medium"
# model_size = "distil-medium.en" # English model (faster)

# Run on GPU with FP16
start_time = time.time()
model = WhisperModel(model_size, device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)
print("Time to load model: ", time.time()-start_time)

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

output_dir = 'recordings'  # Directory to save recordings
output_dir_fast = 'recordings_fast'  # Directory to save recordings

while True:
    try:
        start_time = time.time()
        wav_filename = f"{output_dir}/audio_1.wav"
        segments, info = model.transcribe(wav_filename, beam_size=5, vad_filter=True, language='zh')
        # segments, info = batched_model.transcribe(wav_filename, batch_size=24, temperature=0.0, language='zh') # Better performance
        # print("Time to load model: ", time.time()-start_time)

        # combined_text = " ".join([segment.text for segment in segments])
        

        # print(f"Transcribed text: \n\n")
        combined_text = ""
        for segment in segments:
            combined_text = combined_text + " " + segment.text
            # print(segment.text, end=' ', flush=True)

        print("\033c", end="")
        print(f"Transcribed text ({info.language} | {info.language_probability}): \n\n")
        print(combined_text)

        # for segment in segments:
        #     print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # print("\033c", end="")
        print()

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))