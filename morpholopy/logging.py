import time
import os


class Log:
    def __init__(self, tic, do_message, do_debug):
        self.tic = tic
        self.pid = os.getpid()
        self.do_message = do_message
        self.do_debug = do_debug

    def message(self, *args):
        if self.do_message:
            toc = time.time()
            print(f"[{toc-self.tic:10.2f}] [PID{self.pid}]", *args, flush=True)

    def debug(self, *args):
        if self.do_debug:
            toc = time.time()
            print(f"[{toc-self.tic:10.2f}] [PID{self.pid}] [DEBUG]", *args, flush=True)


class GalaxyLog(Log):
    def __init__(self, tic, do_message, do_debug, galaxy_id):
        super().__init__(tic, do_message, do_debug)
        self.galaxy_id = galaxy_id

    def message(self, *args):
        super().message(f"[galaxy {self.galaxy_id}]", *args)

    def debug(self, *args):
        super().debug(f"[galaxy {self.galaxy_id}]", *args)


class MainLog(Log):
    def __init__(self, log_level="MAIN"):
        tic = time.time()
        self.log_level = log_level
        do_message = log_level != "NONE"
        do_debug = log_level in ["DEBUG", "WORKERDEBUG"]
        super().__init__(tic, do_message, do_debug)

    def get_galaxy_log(self, galaxy_id):
        return GalaxyLog(
            self.tic,
            self.log_level == "WORKER",
            self.log_level == "WORKERDEBUG",
            galaxy_id,
        )
