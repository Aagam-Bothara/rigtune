"""Scripts rigtune runs in an engine's own Python to ask it what it offers (rigtune.backends.facts).

They are files, not code rigtune imports: each runs as `<engine python> <script>`, imports the engine, and prints
what it found. They use the standard library and the engine only, since rigtune is usually not installed in the
engine's environment.
"""
