#!/usr/bin/env python
import os
import platform
import argparse
import datetime
import subprocess
import numpy as np
import torch
import whisper
import pyannote.audio
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import wave
import contextlib

TORCH_DEVICE = "cuda"

if platform.system() == "Darwin":
    TORCH_DEVICE = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print("Using MPS")
    print(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb", device=torch.device(TORCH_DEVICE)
)


def diarize(
    input_file, output_file, num_speakers=2, language="English", model_size="small"
):
    # num_speakers = 2  # @param {type:"integer"}

    # language = "English"  # @param ['any', 'English']

    # model_size = "small"  # @param ['tiny', 'base', 'small', 'medium', 'large']

    model_name = model_size
    if language == "English" and model_size != "large":
        model_name += ".en"

    path = input_file

    if path[-3:] != "wav":
        subprocess.call(["ffmpeg", "-i", path, "audio.wav", "-y"])
        path = "audio.wav"

    model = whisper.load_model(model_size)

    result = model.transcribe(path)
    segments = result["segments"]

    with contextlib.closing(wave.open(path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    audio = Audio()

    def segment_embedding(segment):
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        return embedding_model(waveform[None])

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)

    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    f = open(output_file, "w")

    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write(
                "\n" + segment["speaker"] + " " + str(time(segment["start"])) + "\n"
            )
        f.write(segment["text"][1:] + " ")
    f.close()


if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Diarization program")

    # Add command line arguments
    parser.add_argument(
        "-i", "--input_file", required=True, help="Path to the input file"
    )
    parser.add_argument(
        "-o", "--output_file", required=True, help="Path to the output file"
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Call diarization function with provided arguments
    diarize(args.input_file, args.output_file)
