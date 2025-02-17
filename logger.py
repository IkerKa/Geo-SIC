import logging
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.traceback import install
from datetime import datetime
import json
import os
import time

# Instalar Rich Traceback para mejorar los errores
install()

class Logger:
    def __init__(self, log_file=None, config_path=None):
        self.console = Console()
        self.log_file = log_file
        self.icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
        }
        self.styles = {
            "info": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "success": "bold blue",
        }
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path):
        """Load configuration for styles and icons."""
        with open(config_path, "r") as f:
            config = json.load(f)
            self.icons.update(config.get("icons", {}))
            self.styles.update(config.get("styles", {}))

    def _get_timestamp(self):
        """Get a formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, message, style, level, icon=None):
        """Internal logging method to handle all log types."""
        timestamp = self._get_timestamp()
        icon = icon or self.icons.get(level, "")
        formatted_message = Text.assemble(
            (f"{timestamp} ", "dim"),
            (f"[{level.upper()}] ", style),
            (f"{icon} ", style),
            (message, style),
        )
        self.console.print(formatted_message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp} [{level.upper()}] {message}\n")

    def info(self, message):
        """Log an informational message."""
        self._log(message, self.styles["info"], "info")
    
    def warning(self, message):
        """Log a warning message."""
        self._log(message, self.styles["warning"], "warning")
    
    def error(self, message):
        """Log an error message."""
        self._log(message, self.styles["error"], "error")
    
    def success(self, message):
        """Log a success message."""
        self._log(message, self.styles["success"], "success")
    
    def custom(self, message, color, icon=""):
        """Log a custom message with a specified color and icon."""
        self._log(message, f"bold {color}", "custom", icon)

    def divider(self, text="", style="dim"):
        """Print a divider for log sections."""
        if text:
            self.console.rule(Text(f" {text} ", style=style))
        else:
            self.console.rule(style=style)

    def panel(self, message, title="LOG", style="bold blue"):
        """Log a message inside a Rich Panel."""
        self.console.print(Panel(message, title=title, style=style))

    def banner(self, message, color="blue"):
        """Display a banner-style message."""
        centered_message = Text(f" {message} ", style=f"bold {color}")
        self.console.print(centered_message, style=f"bold {color}", justify="center")
        
    def msg_to_file(self, message, file_path):
        """Write a message to a file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"{timestamp} {message}\n"
        
        #if the file already exists, append the message
        if os.path.exists(file_path):
            with open(file_path, "a") as f:
                f.write(formatted_message)
        else:
            with open(file_path, "w") as f:
                f.write(formatted_message)

    def progress(self, task_message, total=100):
        """Show a progress bar for a task."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task(task_message, total=total)
            for _ in range(total):
                progress.advance()

    def highlight_keywords(self, message, keywords, color="yellow"):
        """Highlight keywords in a message with a specified color."""
        text = Text(message)
        for keyword in keywords:
            text.highlight_words(keyword, style=f"bold {color}")
        
        timestamp = self._get_timestamp()
        formatted_message = Text.assemble(
            (f"{timestamp} ", "dim"),
            text,
        )
        self.console.print(formatted_message)