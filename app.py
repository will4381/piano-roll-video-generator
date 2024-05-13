import os
import librosa
import numpy as np
import moviepy.editor as mpy
from music21 import note, stream, duration
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

class SheetMusicGenerator:
    def __init__(self, song_path, output_path):
        self.song_path = song_path
        self.output_path = output_path
        self.notes = None
        self.timings = None
        self.stream = None

    def note_to_midi(self, note_name):
        return librosa.note_to_midi(note_name) if note_name is not None else None

    def midi_to_note(self, midi_number):
        return librosa.midi_to_note(midi_number) if midi_number is not None else None

    def analyze_audio(self):
        # Load the audio file
        y, sr = librosa.load(self.song_path)

        # Perform harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Perform pitch detection on the harmonic component
        pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)

        # Define a threshold for the minimum magnitude
        threshold = np.median(magnitudes) * 1.5  # Adjust this value to filter more or fewer notes

        # Filter out the positions where the magnitude is too low
        max_magnitude_indices = magnitudes.argmax(axis=0)
        confident_indices = (magnitudes.max(axis=0) > threshold)

        # Initialize notes array
        self.notes = []

        # Iterate over confident indices to get the pitch values
        for j, (mag, idx) in enumerate(zip(magnitudes.max(axis=0), max_magnitude_indices)):
            if confident_indices[j]:  # Only consider frames with magnitudes above the threshold
                pitch = pitches[idx, j]
                if np.isfinite(pitch) and pitch > 0:
                    note_name = librosa.hz_to_note(pitch)
                    self.notes.append(note_name)
                else:
                    self.notes.append(None)  # Placeholder for non-notes or rest
            else:
                self.notes.append(None)  # Placeholder for non-notes or rest

        # Convert notes to MIDI numbers
        midi_notes = [self.note_to_midi(note) for note in self.notes]

        # Replace None values with a placeholder for the median filter
        placeholder = -1
        midi_notes_with_placeholder = [note if note is not None else placeholder for note in midi_notes]

        # Apply a median filter to smooth out rapid note changes
        smoothed_midi_notes = median_filter(midi_notes_with_placeholder, size=5, mode='constant', cval=placeholder)

        # Replace placeholders back to None
        smoothed_midi_notes = [note if note != placeholder else None for note in smoothed_midi_notes]

        # Convert MIDI numbers back to note names
        self.notes = [self.midi_to_note(note) for note in smoothed_midi_notes]

        # Get the timing of each note
        self.timings = librosa.frames_to_time(np.arange(len(self.notes)), sr=sr)

    def transcribe_notes(self):
        # Create a music21 stream
        self.stream = stream.Stream()

        # Iterate over the notes and timings
        for note_name, start_time, end_time in zip(self.notes[:-1], self.timings[:-1], self.timings[1:]):
            if note_name:  # This is a note
                # Correct accidental symbols for compatibility with music21
                note_name = note_name.replace('♯', '#').replace('♭', '-')

                # Parse the note name into a music21 Note
                current_note = note.Note(note_name)
            else:  # This is a rest
                current_note = note.Rest()

            # Calculate the note duration based on the timings
            duration_quarter_length = max(0.25, (end_time - start_time) / 0.25)  # Ensure minimum duration of 0.25
            current_note.duration = duration.Duration(duration_quarter_length)

            # Add the note or rest to the stream
            self.stream.append(current_note)

    def generate_sheet_music_image(self):
        # Create a figure for the sheet music
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the music21 stream as a matplotlib figure
        self.stream.plot('matplotlib', ax=ax)
        
        # Convert the plot to a numpy array
        sheet_music_image = mplfig_to_npimage(fig)

        return fig, ax, sheet_music_image

    def generate_video(self, use_original_audio=True):
        # Generate the sheet music image
        fig, ax, sheet_music_image = self.generate_sheet_music_image()

        # Function to update the sheet music image for each frame
        def make_frame(t):
            # Calculate the frame index based on time
            frame_idx = int(t * len(self.notes) / self.timings[-1])

            # Clear previous highlights
            ax.clear()

            # Re-display the sheet music image
            ax.imshow(sheet_music_image)
            ax.axis('off')

            # Highlight the current note
            if self.notes[frame_idx]:
                ax.text(0.5, 0.1, self.notes[frame_idx], fontsize=24, color='red', ha='center', transform=ax.transAxes)

            # Convert the plot to a numpy array
            return mplfig_to_npimage(fig)

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
        final_clip.write_videofile(self.output_path, fps=24)

    def generate_tutorial(self, use_original_audio=True):
        self.analyze_audio()
        self.transcribe_notes()
        self.generate_video(use_original_audio)

# Example usage
song_path = '24.mp3'
output_path = 'video.mp4'

generator = SheetMusicGenerator(song_path, output_path)
generator.generate_tutorial(use_original_audio=True)
