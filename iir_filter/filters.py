import numpy as np
from scipy.signal import butter, lfilter
from typing import Tuple


def butter_bandpass(
    fs: float, lowcut: float = None, highcut: float = None, order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Butterworth digital and analog filter design.
    Design Nth-order digital or analog Butterworth filter and return the filter coefficients.
    Args:
         fs: The sampling frequency of the digital system
         lowcut: lowpass frequency
         highcut: highpass frequency
         order: the order of filter
    Returns:
        b: numerator polynomials of the filter
        a: denominator polynomials of the filter
    """
    nyq = 0.5 * fs
    if lowcut is None:
        b, a = butter(order, highcut / nyq, btype="lowpass")
    elif highcut is None:
        b, a = butter(order, lowcut / nyq, btype="highpass")
    else:
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def butter_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = None,
    highcut: float = None,
    order: int = 4,
) -> np.ndarray:
    """
    Filter data along one-dimension with Butterworth filter
    Args:
        signal: An N-dimensional input array.
        fs: The sampling frequency of the digital system
        lowcut: lowpass frequency
        highcut: highpass frequency
        order: the order of filter
    Returns:
        y: The output of the digital filter.
    """
    b, a = butter_bandpass(fs, lowcut, highcut, order=order)
    y = lfilter(b, a, signal)
    return y
