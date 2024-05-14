import os
import json
import numpy as np
import librosa
import moviepy.editor as mpy
import crepe
from scipy.ndimage import median_filter
from moviepy.video.VideoClip import TextClip

class MelodyExtractor:
    def __init__(self, song_path, output_path_json, output_path_video):
        self.song_path = song_path
        self.output_path_json = output_path_json
        self.output_path_video = output_path_video
        self.notes = []
        self.timings = []

    def note_to_midi(self, note_name):
        return librosa.note_to_midi(note_name) if note_name is not None else None

    def midi_to_note(self, midi_number):
        return librosa.midi_to_note(midi_number) if midi_number is not None else None

    def analyze_audio(self):
        # Load the audio file
        y, sr = librosa.load(self.song_path, sr=None)

        # Extract melody using the Melodia algorithm
        melody = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

        pitches, magnitudes = melody
        times = librosa.times_like(pitches)

        # Filter out the unvoiced pitches
        voiced_pitches = pitches[~np.isnan(pitches)]
        voiced_times = times[~np.isnan(pitches)]

        # Convert frequencies to MIDI notes
        midi_notes = librosa.hz_to_midi(voiced_pitches)

        # Smooth the midi notes using a median filter
        midi_notes = median_filter(midi_notes, size=3)

        # Group simultaneous notes (chords)
        grouped_notes = []
        current_group = [midi_notes[0]]
        current_time = voiced_times[0]
        time_threshold = 0.05  # Threshold to group notes into chords

        for midi_note, t in zip(midi_notes[1:], voiced_times[1:]):
            if t - current_time < time_threshold:
                current_group.append(midi_note)
            else:
                grouped_notes.append((current_group, current_time))
                current_group = [midi_note]
                current_time = t

        if current_group:
            grouped_notes.append((current_group, current_time))

        self.notes = []
        self.timings = []

        for group, t in grouped_notes:
            self.notes.append([self.midi_to_note(note) for note in group])
            self.timings.append(t)

    def generate_json(self):
        notes_data = []
        for i, note_group in enumerate(self.notes[:-1]):
            note_info = {
                "notes": note_group,
                "start_time": self.timings[i],
                "duration": self.timings[i+1] - self.timings[i]
            }
            notes_data.append(note_info)

        # Handle the last note group separately
        last_note_info = {
            "notes": self.notes[-1],
            "start_time": self.timings[-1],
            "duration": self.timings[-1] - self.timings[-2] if len(self.timings) > 1 else 0
        }
        notes_data.append(last_note_info)

        with open(self.output_path_json, 'w') as outfile:
            json.dump(notes_data, outfile, indent=4)

    def generate_video(self, use_original_audio=True):
        # Function to create a text clip for the notes
        def make_frame(t):
            frame_idx = int(t * len(self.notes) / self.timings[-1])
            note_text = " + ".join(self.notes[frame_idx]) if self.notes[frame_idx] else ""
            text_clip = TextClip(note_text, fontsize=70, color='white', bg_color='black', size=(1920, 1080))
            return text_clip.get_frame(t)

        # Create the video clip using the make_frame function
        animation = mpy.VideoClip(make_frame, duration=self.timings[-1])

        if use_original_audio:
            # Use the original audio
            audio = mpy.AudioFileClip(self.song_path)
        else:
            # Generate silence (dummy audio clip if needed)
            audio = mpy.AudioClip(lambda t: np.sin(440 * 2 * np.pi * t), duration=self.timings[-1])

        # Combine the video and audio
        final_clip = animation.set_audio(audio)

        # Write the final video
        final_clip.write_videofile(self.output_path_video, fps=24)

    def generate_tutorial(self):
        self.analyze_audio()
        self.generate_json()
        self.generate_video(use_original_audio=True)

# Example usage
song_path = '24.mp3'
output_path_json = 'notes.json'
output_path_video = 'notes_video.mp4'

extractor = MelodyExtractor(song_path, output_path_json, output_path_video)
extractor.generate_tutorial()
