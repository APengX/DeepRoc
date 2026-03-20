import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deeproc.core import run_iteration


class DeepRocTests(unittest.TestCase):
    def make_repo(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="deeproc-test-"))
        subprocess.run(["git", "init"], cwd=tmp, check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "DeepRoc Test"], cwd=tmp, check=True, capture_output=True)
        return tmp

    def write_lab(self, root: Path, initial_message: str, proposal_message: str) -> Path:
        (root / "program.md").write_text("Improve message.txt to maximize score.\n")
        (root / "message.txt").write_text(initial_message)
        (root / "score.py").write_text(
            "from pathlib import Path\n"
            "message = Path('message.txt').read_text()\n"
            "print(f\"score: {message.count('agent')}\")\n"
        )
        (root / "proposal.json").write_text(
            json.dumps(
                {
                    "summary": "Update message",
                    "rationale": "Replay provider update.",
                    "changes": [
                        {"action": "write", "path": "message.txt", "content": proposal_message}
                    ],
                }
            )
        )
        (root / "deeproc.json").write_text(
            json.dumps(
                {
                    "name": "test-lab",
                    "program_path": "program.md",
                    "mutable_paths": ["message.txt"],
                    "provider": {"type": "replay", "path": "proposal.json"},
                    "evaluation": {
                        "command": ["python3", "score.py"],
                        "metric_pattern": "score:\\s*([0-9.]+)",
                        "goal": "max",
                        "timeout_seconds": 10,
                    },
                }
            )
        )
        subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True)
        return root / "deeproc.json"

    def test_keeps_improvement(self) -> None:
        root = self.make_repo()
        config_path = self.write_lab(root, "hello\n", "agent agent agent\n")
        record = run_iteration(config_path)
        self.assertEqual(record.status, "keep")
        self.assertTrue(record.kept)
        self.assertAlmostEqual(record.metric or 0.0, 3.0)
        self.assertEqual((root / "message.txt").read_text(), "agent agent agent\n")
        state = json.loads((root / ".deeproc" / "state.json").read_text())
        self.assertEqual(state["best_metric"], 3.0)

    def test_discards_regression(self) -> None:
        root = self.make_repo()
        config_path = self.write_lab(root, "agent agent\n", "hello\n")
        state_dir = root / ".deeproc"
        state_dir.mkdir()
        (state_dir / "state.json").write_text(json.dumps({"best_metric": 2.0}) + "\n")
        record = run_iteration(config_path)
        self.assertEqual(record.status, "discard")
        self.assertFalse(record.kept)
        self.assertEqual((root / "message.txt").read_text(), "agent agent\n")
        state = json.loads((root / ".deeproc" / "state.json").read_text())
        self.assertEqual(state["best_metric"], 2.0)


if __name__ == "__main__":
    unittest.main()
