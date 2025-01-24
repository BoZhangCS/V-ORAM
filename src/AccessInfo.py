# Structure to store access information
class AccessInfo:
    def __init__(self, down_sync=0, down_async=0, up_sync=0, up_async=0, rtt_sync=0, rtt_async=0,
                 evict_record_flag=False):
        self.down_sync = down_sync
        self.down_async = down_async
        self.up_sync = up_sync
        self.up_async = up_async
        self.rtt_sync = rtt_sync
        self.rtt_async = rtt_async
        self.evict_record_flag = evict_record_flag

    def clear(self):
        self.down_sync = 0
        self.down_async = 0
        self.up_sync = 0
        self.up_async = 0
        self.rtt_sync = 0
        self.rtt_async = 0
        self.evict_record_flag = False
