import logging
import json
import datetime as st
from typing import override


logger = logging.getLogger("debate_logger")

class MyJSONFormatter(logging.Formatter):
    def __init__(self, fmt_keys: dict):
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}
        super().__init__()
        def format(self, record:logging.LogRecord):
            message = self._prepare_log_dict(record)
            return json.dumps(message, default=str)
    @override
    def _prepare_log_dict(self, record:logging.LogRecord):
        always_include = {
            "message": record.getMessage(),
            "timestamp": st.datetime.fromtimestamp(record.created, tz = st.timezone.idt).isoformat(),
        }
        message = {
            key: msg_val
            if (msg_val := always_include.pop(val,None) is not None)
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_include)
        return message

class MyHTMLFormatter(logging.Formatter):
    def __init__(self, fmt_keys: dict):
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}
        super().__init__()
    
    def format(self, record: logging.LogRecord):
        # Simple HTML format
        message = record.getMessage()
        return f"<div>{message}</div>"

# Extras are expected to be: speaker, receiver, message_type


