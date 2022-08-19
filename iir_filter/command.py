import pandas as pd
from otlang.sdk.syntax import Keyword, Positional, OTLType
from pp_exec_env.base_command import BaseCommand, Syntax
from .filters import butter_bandpass_filter


class IirFilterCommand(BaseCommand):
    """
    Filter data along one-dimension with Butterworth filter
    iir_filter signal fs=100, lowcut=3, highcut=10, order=4
    """
    syntax = Syntax(
        [
            Positional("signal", required=True, otl_type=OTLType.TEXT),
            Keyword("fs", required=True, otl_type=OTLType.NUMERIC),
            Keyword("lowcut", required=True, otl_type=OTLType.NUMERIC),
            Keyword("highcut", required=True, otl_type=OTLType.NUMERIC),
            Keyword("order", required=False, otl_type=OTLType.INTEGER)
        ],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_progress('Start iir_filter command')
        signal_name = self.get_arg("signal").value
        fs = self.get_arg("fs").value
        lowcut = self.get_arg("lowcut").value
        highcut = self.get_arg("highcut").value
        order = self.get_arg("order").value or 4

        raw_signal = df[signal_name].values
        filtered_signal = butter_bandpass_filter(raw_signal, fs=fs, lowcut=lowcut, highcut=highcut, order=order)
        df[f"filtered_{signal_name}"] = filtered_signal
        self.log_progress('First part is complete.', stage=1, total_stages=1)

        return df
