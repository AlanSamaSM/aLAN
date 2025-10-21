
import torch
import torchaudio
from csm_main.generator import load_csm_1b

def main():
    """
    Your everyday companion.

    This script uses the CSM model to generate speech from your text.
    """

    # Select the best available device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    generator = load_csm_1b(device)

    # Get user input
    text = input("What should I say? ")

    # Generate audio
    audio_tensor = generator.generate(
        text=text,
        speaker=0,  # Use a default speaker
        max_audio_length_ms=10_000,
    )

    # Save audio
    output_filename = "output.wav"
    torchaudio.save(
        output_filename,
        audio_tensor.unsqueeze(0).cpu(),
        generator.sample_rate
    )
    print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    main()
