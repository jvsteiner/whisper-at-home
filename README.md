# Whisper at Home

There are tons of new code repos on how to run whisper from OpenAI. My use case involved doing transcriptions of interviews locally on an M1 Macbook Pro, with diarization.

I used the excellent colab notebook [found here](https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing) as the basis, and found that I could make it work pretty well.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -q git+https://github.com/openai/whisper.git
pip install -q git+https://github.com/pyannote/pyannote-audio
```

## Usage

```bash
env PYTORCH_ENABLE_MPS_FALLBACK=1 ./python diarize.py -i <input_file> -o <output_file>
```
