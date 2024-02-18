from llama_index.core.tools import FunctionTool
import os

note_file = os.path.join("data", "note.txt")

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w").close()

    with open(note_file, "a") as f:
        f.writelines([note + "\n"])

    return "note saved"

note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="This tool saves a note to a file.",
)