from pathlib import Path


message = Path("message.txt").read_text()
score = message.count("agent")
print(f"score: {score}")
