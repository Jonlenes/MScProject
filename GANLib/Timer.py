import time


class Timer(object):
    def start(self):
        self.start_time = time.time()
    
    
    def diff(self):
        return self.elapsed_time()
        
    
    def elapsed_time(self):
        sec = time.time() - self.start_time
        if sec < 60:
            return str(round(sec, 2)) + " sec"
        elif sec < (60 * 60):
            return str(round(sec / 60, 2)) + " min"
        else:
            return str(round(sec / (60 * 60), 2)) + " hr"