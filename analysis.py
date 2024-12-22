import pyaudio
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from pychord import find_chords_from_notes

def find_closest_note(frequency):
    """Find the closest musical note to the given frequency."""
    A4 = 440  # Frequency of A4
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    if frequency == 0:
        return None

    semitones = 12 * np.log2(frequency / A4)
    closest_note_index = int(round(semitones)) % 12

    return NOTES[closest_note_index]

def list_audio_devices():
    """List all available audio input devices."""
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"[{i}] {info['name']}")
    p.terminate()

def analyze_chords(input_device_index=None):
    """Analyze the audio input and identify chords."""
    CHUNK = 8192  # Larger chunk for better frequency resolution
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    ZP_FACTOR = 4
    AGGREGATE_FRAMES = 10  # Number of frames to average

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK)

    print("Chord detector is listening...")

    try:
        frame_data = []  # To hold aggregated frames
        while True:
            # Read audio data
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

            # Apply window and zero-padding
            windowed_data = data * np.blackman(len(data))
            padded_data = np.pad(windowed_data, (0, len(windowed_data) * (ZP_FACTOR - 1)), 'constant')

            # Perform FFT
            fft = rfft(padded_data)
            frequencies = rfftfreq(len(padded_data), d=1/RATE)
            magnitudes = np.abs(fft)

            # Aggregate multiple frames for stability
            frame_data.append(magnitudes)
            if len(frame_data) < AGGREGATE_FRAMES:
                continue
            avg_magnitudes = np.mean(frame_data, axis=0)
            frame_data.pop(0)  # Remove oldest frame

            # Find significant peaks in the spectrum
            peak_indices, _ = find_peaks(avg_magnitudes, height=np.max(avg_magnitudes) * 0.1, distance=20)
            peak_frequencies = frequencies[peak_indices]

            # Filter harmonics and map peaks to notes
            detected_notes = []
            for freq in peak_frequencies:
                if 20 < freq < 5000:  # Focus on musical range
                    note = find_closest_note(freq)
                    if note:
                        detected_notes.append(note)

            # Use pychord to identify the chord
            try:
                possible_chords = find_chords_from_notes(list(set(detected_notes)))
                print(f"Detected Notes: {list(set(detected_notes))}")
                if possible_chords:
                    print(f"Possible Chords: {', '.join(str(chord) for chord in possible_chords)}")
                else:
                    print("Detected Chord: Unknown")
            except Exception as e:
                print(f"Error detecting chord: {e}")

    except KeyboardInterrupt:
        print("Chord detection stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    print("Listing available audio devices...")
    list_audio_devices()
    device_index = int(input("Enter the device index for system audio: "))
    analyze_chords(input_device_index=device_index)
