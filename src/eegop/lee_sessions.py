"""Session-preserving access to Lee2019_MI (OpenBMI) through MOABB 1.5.

In MOABB 1.5.0, ``Lee2019_MI._get_single_subject_data`` stores recording
session s under the key ``str(s - 1)``, while the session filter applied by
the base dataset keeps keys that match the requested session numbers. With the
default ``sessions=(1, 2)`` only key ``'1'`` survives, i.e. only the second
recording session is returned, and requesting ``sessions=(2,)`` returns
nothing. This subclass relabels the keys with the true session numbers so both
sessions are returned and identified correctly.
"""

from moabb.datasets import Lee2019_MI


class Lee2019MISessions(Lee2019_MI):
    """Lee2019_MI with session keys "1" and "2" for the two recording sessions."""

    def _get_single_subject_data(self, subject):
        data = super()._get_single_subject_data(subject)
        return {str(int(k) + 1): v for k, v in data.items()}
