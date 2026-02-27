# Aliya TTS Worker (Qwen3-TTS)

Dedicated worker service for high-fidelity voice synthesis using **Qwen3-TTS**. Designed to offload computation to GPU-enabled machines.

## Features
- State-of-the-art voice cloning based on Qwen3-TTS.
- GPU acceleration support (CUDA).
- REST API for remote synthesis.
- Lightweight and easy to deploy.

## Quick Start (Ubuntu/Windows with GPU)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/argokz/tts-worker.git
   cd tts-worker
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the worker**:
   ```bash
   python main.py
   ```

The worker will start on port `8005` by default.

## API Usage
`POST /synthesize`
- `text`: Prompt to synthesize.
- `language`: Language code (default: `ru`).
- `reference_audio`: WAV file for voice cloning.

## License
MIT
