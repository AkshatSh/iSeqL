import os
import sys
import threading
import time

from progress_thread import (
    ProgressableThread,
)

from typing import (
    List,
    Dict,
    Tuple,
)


class ThreadManager(object):
    def __init__(self):
        self.thread_data : Dict[int, ProgressableThread] = {}
    
    def add_thread(self, thread: ProgressableThread) -> int:
        thread.daemon = True  
        thread.start()
        t_id = thread.get_thread_id()
        self.thread_data[t_id] = thread
        return t_id
    
    def get_progress(self, t_id: int) -> Dict[str, object]:
        print(self.thread_data)
        if t_id in self.thread_data:
            return self.thread_data[t_id].get_progress()
        else:
            return {"Error": "Not Found"}
        
    def get_thread(self, t_id: int) -> threading.Thread:
        return self.thread_data[t_id]
    
    def get_all_thread_ids(self) -> List[int]:
        return list(self.thread_data.keys())
    
    def remove_thread(self, t_id: int) -> bool:
        del self.thread_data[t_id]
        return True


class ThreadManagerThread(threading.Thread):
    def set_threading_manager(self, thread_manager: ThreadManager):
        self.thread_manager = thread_manager

    def run(self):
        while True:
            for t_id in self.thread_manager.get_all_thread_ids():
                thread_t = self.thread_manager.get_thread(t_id)
                if not thread_t.isAlive():
                    self.thread_manager.remove_thread(t_id)
            
            # run this every 10 seconds
            time.sleep(10)

def start_thread_manager() -> Tuple[ThreadManager, threading.Thread]:
    tm = ThreadManager()
    tm_thread = None
    thread = ThreadManagerThread(name = "Thread_Manager_Thread")
    thread.set_threading_manager(tm)
    thread.daemon = True
    thread.start()
    return tm, thread

