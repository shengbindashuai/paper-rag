"""目录变更通知：Windows 原生事件，其他系统使用 watchdog 原生观察器。"""
import os
from threading import Event, Thread


class DirectoryEvents:
    def __init__(self, folders, callback):
        self.folders, self.callback = folders, callback
        self.ready = Event()
        self.error = None
        self.stop_handle = None
        self.thread = Thread(target=self.run, name='knowledge-file-events', daemon=True)

    def start(self):
        if os.name != 'nt':
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            owner = self

            class Handler(FileSystemEventHandler):
                def __init__(self, reason):
                    self.reason = reason

                def on_any_event(self, event):
                    if event.event_type in {'created', 'modified', 'moved', 'deleted'}:
                        try:
                            owner.callback(self.reason)
                        except Exception as exc:
                            owner.error = exc
                            owner.callback('watcher_error')

            self.thread = Observer()
            for folder, reason in self.folders:
                self.thread.schedule(Handler(reason), str(folder), recursive=True)
            self.thread.start()
            self.ready.set()
            return
        import win32event
        self.stop_handle = win32event.CreateEvent(None, True, False, None)
        self.thread.start()
        self.ready.wait()
        if self.error:
            self.close()
            raise self.error

    def run(self):
        import win32con
        import win32event
        import win32file
        handles = []
        try:
            flags = (win32con.FILE_NOTIFY_CHANGE_FILE_NAME | win32con.FILE_NOTIFY_CHANGE_DIR_NAME |
                     win32con.FILE_NOTIFY_CHANGE_SIZE | win32con.FILE_NOTIFY_CHANGE_LAST_WRITE)
            for folder, _ in self.folders:
                handles.append(win32file.FindFirstChangeNotification(str(folder), True, flags))
            self.ready.set()
            while True:
                event = win32event.WaitForMultipleObjects([self.stop_handle, *handles], False, win32event.INFINITE)
                position = event - win32event.WAIT_OBJECT_0
                if position == 0:
                    break
                if not 1 <= position <= len(handles):
                    raise RuntimeError('目录通知等待失败')
                win32file.FindNextChangeNotification(handles[position - 1])
                self.callback(self.folders[position - 1][1])
        except Exception as exc:
            self.error = exc
            self.ready.set()
            self.callback('watcher_error')
        finally:
            for handle in handles:
                win32file.FindCloseChangeNotification(handle)

    def close(self):
        if os.name != 'nt':
            if self.thread.ident is not None:
                self.thread.stop()
                self.thread.join()
            return
        if self.stop_handle is not None:
            import win32api
            import win32event
            win32event.SetEvent(self.stop_handle)
            self.thread.join()
            win32api.CloseHandle(self.stop_handle)
            self.stop_handle = None
