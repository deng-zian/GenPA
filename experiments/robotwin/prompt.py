class EpisodePrompt:
    """Keep only the current episode's text context, never a disk cache."""

    def __init__(self, encode):
        self.encode = encode
        self.reset()

    def reset(self):
        self.values = {}
        self.calls = 0

    def get(self, instruction):
        if instruction not in self.values:
            self.values[instruction] = self.encode(instruction)
            self.calls += 1
        return self.values[instruction]
