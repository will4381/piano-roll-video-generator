import os
import json
import numpy as np
import librosa
import crepe
from scipy.ndimage import median_filter

class MelodyExtractor:
    def __init__(self, song_path, output_path):
        self.song_path = song_path
        self.output_path = output_path
        self.notes = []
        self.timings = []

    def note_to_midi(self, note_name):
        return librosa.note_to_midi(note_name) if note_name is not None else None

    def midi_to_note(self, midi_number):
        return librosa.midi_to_note(midi_number) if midi_number is not None else None

    def analyze_audio(self):
        # Load the audio file
        y, sr = librosa.load(self.song_path, sr=16000)

        # Harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Use CREPE to analyze the harmonic component and extract pitch
        time, frequency, confidence, _ = crepe.predict(y_harmonic, sr, viterbi=True, step_size=100)

        # Threshold for pitch confidence
        confidence_threshold = 0.8

        # Extract pitches above the confidence threshold
        pitches = frequency[confidence > confidence_threshold]
        times = time[confidence > confidence_threshold]

        # Convert frequencies to MIDI notes
        midi_notes = librosa.hz_to_midi(pitches)

        # Smooth the midi notes using a median filter
        midi_notes = median_filter(midi_notes, size=3)

        # Filter out improbable notes (non-melodic components)
        unique, counts = np.unique(midi_notes, return_counts=True)
        note_frequencies = dict(zip(unique, counts))
        melody_threshold = np.median(list(note_frequencies.values()))

        melody_notes = [note for note in midi_notes if note_frequencies[note] > melody_threshold]

        self.notes = []
        self.timings = []

        for midi_note, t in zip(melody_notes, times):
            self.notes.append(self.midi_to_note(midi_note))
            self.timings.append(t)

    def generate_json(self):
        notes_data = []
        for i, note_name in enumerate(self.notes[:-1]):
            note_info = {
                "note": note_name,
                "start_time": self.timings[i],
                "duration": self.timings[i+1] - self.timings[i]
            }
            notes_data.append(note_info)

        # Handle the last note separately
        last_note_info = {
            "note": self.notes[-1],
            "start_time": self.timings[-1],
            "duration": self.timings[-1] - self.timings[-2] if len(self.timings) > 1 else 0
        }
        notes_data.append(last_note_info)

        with open(self.output_path, 'w') as outfile:
            json.dump(notes_data, outfile, indent=4)

    def generate_tutorial(self):
        self.analyze_audio()
        self.generate_json()

# Example usage
song_path = '24.mp3'
output_path = 'notes.json'

extractor = MelodyExtractor(song_path, output_path)
extractor.generate_tutorial()