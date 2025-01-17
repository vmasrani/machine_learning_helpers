
import shutil
import sys


class SplitScreen:
    def __init__(self, left_ratio=0.85):
        # Get terminal width
        self.term_width = shutil.get_terminal_size().columns
        self.term_height = shutil.get_terminal_size().lines
        # Calculate panel widths
        self.left_width = int(self.term_width * left_ratio)
        self.right_width = self.term_width - self.left_width
        # Keep track of lines printed in each panel
        self.left_lines = 0
        self.right_lines = 0
    def print_panel(self, text, panel='left'):
        # Save cursor position
        sys.stdout.write('\033[s')
        if panel == 'left':
            # Move to appropriate position in left panel
            sys.stdout.write(f'\033[{self.left_lines + 1};0H')
            # Wrap text to panel width
            wrapped_text = self._wrap_text(text, self.left_width)
            sys.stdout.write(wrapped_text)
            self.left_lines += wrapped_text.count('\n') + 1
        else:  # right panel
            # Move to appropriate position in right panel
            sys.stdout.write(f'\033[{self.right_lines + 1};{self.left_width}H')
            # Wrap text to panel width
            wrapped_text = self._wrap_text(text, self.right_width)
            sys.stdout.write(wrapped_text)
            self.right_lines += wrapped_text.count('\n') + 1
        # Restore cursor position
        sys.stdout.write('\033[u')
        sys.stdout.flush()
    def _wrap_text(self, text, width):
        # Simple text wrapping
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        lines.append(' '.join(current_line))
        return '\n'.join(lines)

