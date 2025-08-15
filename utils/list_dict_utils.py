import time
from itertools import islice


def to_change_logs(d: dict):
    return {int(t): v for t, v in d.items()}


def sorted_dict(d: dict, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[0], reverse=reverse))


def sort_log(log):
    log = to_change_logs(log)
    log = sorted_dict(log)
    return log


def sort_log_dict(log_dict):
    for key in log_dict:
        log = log_dict[key]
        log_dict[key] = sort_log(log)
    return log_dict


def cut_change_logs(
    change_logs: dict,
    end_time: int = None,
    start_time: int = None,
    duration: int = None,
    alt_value=None,
):
    if not end_time:
        end_time = int(time.time())

    if not start_time:
        if not duration:
            raise ValueError("start_time or duration must be set")
        else:
            start_time = end_time - duration

    change_logs = to_change_logs(change_logs)
    change_logs = sorted_dict(change_logs)
    for t in change_logs.keys():
        if (t < start_time) or (t > end_time):
            change_logs[t] = alt_value

    return change_logs


def chunks(input_list: list, size: int):
    for i in range(0, len(input_list), size):
        yield input_list[i : i + size]


def chunks_dict(data: dict, size=50):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}
