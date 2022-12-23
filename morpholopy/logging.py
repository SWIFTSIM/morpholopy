#!/usr/bin/env python3

"""
logging.py

Custom logging mechanism, useful for making print()
statements a bit more informative, and for easily controlling
what gets printed.
"""

import time
import os


class Log:
    """
    Abstract parent class for all logging objects.

    A logging object always prints a time stamp and
    the process ID of the active (sub)process. Two types
    of messages can be printed using either message() or
    debug(). Flags control which of these functions actually
    does something.
    """

    tic: float
    pid: int
    do_message: bool
    do_debug: bool

    def __init__(self, tic: float, do_message: bool, do_debug: bool):
        self.tic = tic
        self.pid = os.getpid()
        self.do_message = do_message
        self.do_debug = do_debug

    def message(self, *args):
        """
        Normal message (the useful equivalent of print()).
        """
        if self.do_message:
            toc = time.time()
            print(f"[{toc-self.tic:10.2f}] [PID{self.pid}]", *args, flush=True)

    def debug(self, *args):
        """
        Debug message (usually not useful for production runs).
        """
        if self.do_debug:
            toc = time.time()
            print(f"[{toc-self.tic:10.2f}] [PID{self.pid}] [DEBUG]", *args, flush=True)


class GalaxyLog(Log):
    """
    Log specialisation for individual galaxies.
    All log messages also include the galaxy index, to make them more informative.
    """

    galaxy_id: int

    def __init__(self, tic: float, do_message: bool, do_debug: bool, galaxy_id: int):
        super().__init__(tic, do_message, do_debug)
        self.galaxy_id = galaxy_id

    def message(self, *args):
        """
        Overwrite the default message() by also writing the galaxy index.
        """
        super().message(f"[galaxy {self.galaxy_id}]", *args)

    def debug(self, *args):
        """
        Overwrite the default debug() by also writing the galaxy index.
        """
        super().debug(f"[galaxy {self.galaxy_id}]", *args)


class MainLog(Log):
    """
    Normal Log specialisation for the main process. Only one instance of this class
    should be created.

    The log level determines what this Log and GalaxyLog instances spawned
    by get_galaxy_log() are allowed to print.
    """

    log_level: str

    def __init__(self, log_level: str = "MAIN"):
        tic = time.time()
        self.log_level = log_level
        do_message = log_level != "NONE"
        do_debug = log_level in ["DEBUG", "WORKERDEBUG"]
        super().__init__(tic, do_message, do_debug)

    def get_galaxy_log(self, galaxy_id: int):
        """
        Get a GalaxyLog instance for the given galaxy index.

        Use the log level to determine what it is allowed to print.
        """
        return GalaxyLog(
            self.tic,
            self.log_level == "WORKER",
            self.log_level == "WORKERDEBUG",
            galaxy_id,
        )
