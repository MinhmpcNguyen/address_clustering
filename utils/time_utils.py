from collections import Counter
from operator import add


class TimeUtils:
    @staticmethod
    def timestamp_to_hour(timestamp: int | float) -> int:
        """Converts a timestamp to the hour of the day (0–23).

        Args:
            timestamp (int | float): The timestamp in seconds.

        Returns:
            int: The hour of the day (0–23).
        """
        return int(timestamp) % 86400 // 3600

    @staticmethod
    def compute_time_histogram(
        timestamps: int | list[int | float], as_str: bool = False
    ) -> list[int] | str:
        """Computes a 24-hour histogram from a list of timestamps.

        Args:
            timestamps (int | list[int | float]): Either 0 or a list of Unix timestamps.
            as_str (bool): If True, returns the histogram as a string. Defaults to False.

        Returns:
            list[int] | str: 24-length list of hour counts, or its string form.
        """
        if timestamps == 0:
            histogram = [0] * 24
        else:
            hours = list(map(TimeUtils.timestamp_to_hour, timestamps))
            hour_counter = Counter(hours)
            histogram = [hour_counter.get(hour, 0) for hour in range(24)]

        return str(histogram) if as_str else histogram

    @staticmethod
    def get_time_histogram(timestamps: int | list[int | float]) -> list[int]:
        """Wrapper for compute_time_histogram that returns the result as a list.

        Args:
            timestamps (int | list[int | float]): Input timestamps.

        Returns:
            list[int]: Histogram of hour counts.
        """
        return TimeUtils.compute_time_histogram(timestamps, as_str=False)

    @staticmethod
    def get_time_histogram_str(timestamps: int | list[int | float]) -> str:
        """Wrapper for compute_time_histogram that returns the result as a string.

        Args:
            timestamps (int | list[int | float]): Input timestamps.

        Returns:
            str: String representation of the histogram.
        """
        return TimeUtils.compute_time_histogram(timestamps, as_str=True)

    @staticmethod
    def get_time(row_from: list[int], row_to: list[int]) -> list[int | float]:
        """Sums corresponding elements from 'From_time' and 'To_time' in a row.

        Args:
            row (dict): Dictionary with keys 'From_time' and 'To_time'.

        Returns:
            list[int | float]: list of element-wise sums.
        """
        return list(map(add, row_from, row_to))
