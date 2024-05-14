A early implementation of creating piano roll tutorial videos. The melody extraction is quite difficult, as we want the base melody excluding background noise. Current implementation uses CREPE which is basic level ML integration, but to further the melody extraction a RNN based melody extraction method would work much better.

Current implementation generates a .json response of notes pressed with a time stamp relative to the audio file, as well as how long each note is pressed. We also generate a short video using moviepy showing the notes played, with the original audio played in the background.

Further implemnetation would require the addition of animations or a piano simulator of sorts.
