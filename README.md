# Moire-cleaner
This project develops a program that automatically detects and removes moiré patterns, the wave-like interference artifacts that appear when photographing computer or smartphone screens. The goal is to reduce moiré while preserving important details such as text and edges.

The default `main.py` pipeline applies adaptive notch filtering followed by
screen cleaning in areas detected as moiré. Set `INPUT_PATH` near the top of
`main.py` to choose the image to process. The optional `--notch-only` and
`--screen-clean` flags run either method on its own.
