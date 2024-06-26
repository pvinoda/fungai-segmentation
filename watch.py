#!/usr/bin/env python

from pathlib import Path
from watchdog.observers.polling import PollingObserver
from watchdog.events import PatternMatchingEventHandler
import subprocess as sp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--watch","-w",help="Directory to be monitored for new files/folders")
args = parser.parse_args()

# Global Variables
print("Starting watch on " + args.watch + " for new files or directories")
dir_to_watch = Path(args.watch).expanduser().resolve()

# Watchdog Event Handler Syntax
class Handler(PatternMatchingEventHandler):

    def __init__(self,patterns):
        PatternMatchingEventHandler.__init__(self, patterns=patterns,
                                             ignore_directories=False,
                                             ignore_patterns=['.*','*/.*'],
                                             # case_sensitive=False
                                             )

    def on_created(self, event):
    # other options: on_any_event, on_created, on_deleted, on_modified, on_moved
        # print(event.src_path)
        if event.is_directory:
            print(event.src_path + " is a directory")
            # Any other commands you want to do here
        


if __name__ == "__main__":
    event_handler = Handler(patterns='[*]')
    observer = PollingObserver()
    observer.schedule(event_handler, path=dir_to_watch, recursive=True)
    observer.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
