# -*- coding: utf-8 -*-

"""pandas库版本兼容模块"""

import re
import warnings

import pandas as pd


# pandas
PD_VERSION = pd.__version__
PD_VERSION_INFO = tuple(int(x) for x in re.findall(r'\d+', PD_VERSION)[:2])

# 月末频率别名: pandas 2.2 起 'M' 改为 'ME', pandas 3 移除 'M'
MONTH_END_FREQ = 'ME' if PD_VERSION_INFO >= (2, 2) else 'M'
YEAR_END_FREQ = 'YE' if PD_VERSION_INFO >= (2, 2) else 'A'


def normalize_freq(freq):
    """将 'M' / 'Y' / 'A' 等旧的频率别名转换为当前 pandas 版本支持的别名"""
    mapping = {'M': MONTH_END_FREQ, 'ME': MONTH_END_FREQ,
               'Y': YEAR_END_FREQ, 'A': YEAR_END_FREQ, 'YE': YEAR_END_FREQ}
    if isinstance(freq, str):
        return mapping.get(freq.upper(), freq)
    return freq


def stack_keepna(df, level=-1):
    """DataFrame.stack 并保留 NaN 值 (兼容 pandas 1.x ~ 3.x)"""
    if PD_VERSION_INFO >= (3, 0):
        return df.stack(level=level)
    elif PD_VERSION_INFO >= (2, 1):
        return df.stack(level=level, future_stack=True)
    else:
        return df.stack(level=level, dropna=False)


def rolling_apply(
    x,
    window,
    func,
    min_periods=None,
    freq=None,
    center=False,
    args=None,
    kwargs=None
):
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    if PD_VERSION_INFO >= (0, 23):
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).apply(
            func, raw=False, args=args, kwargs=kwargs
        )
    elif PD_VERSION_INFO >= (0, 18):
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).apply(
            func, args=args, kwargs=kwargs
        )
    else:
        return pd.rolling_apply(
            x,
            window,
            func,
            min_periods=min_periods,
            freq=freq,
            center=center,
            args=args,
            kwargs=kwargs
        )


def rolling_mean(x, window, min_periods=None, center=False):
    if PD_VERSION_INFO >= (0, 18):
        return x.rolling(window, min_periods=min_periods, center=center).mean()
    else:
        return pd.rolling_mean(
            x, window, min_periods=min_periods, center=center
        )


def rolling_std(x, window, min_periods=None, center=False, ddof=1):
    if PD_VERSION_INFO >= (0, 18):
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).std(ddof=ddof)
    else:
        return pd.rolling_std(
            x, window, min_periods=min_periods, center=center, ddof=ddof
        )


# statsmodels
with warnings.catch_warnings():
    # 有的版本依赖的 pandas 库会有 deprecated warning
    warnings.simplefilter("ignore")
    import statsmodels
    from statsmodels.api import OLS, qqplot, ProbPlot
    from statsmodels.tools.tools import add_constant
