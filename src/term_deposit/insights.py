"""Turning the data and the models into business answers: who subscribes, and why."""

import pandas as pd
from scipy import stats

from term_deposit.config import TARGET


def subscription_rates(
    df: pd.DataFrame, column: str, sort: bool = True, target: str = TARGET
) -> pd.DataFrame:
    """Customers, subscribers and subscription rate for each value of `column`.

    `low` and `high` bound a 95% Wilson interval: the range the true rate plausibly lies in,
    given how many customers the group has. Small groups get wide intervals, so a striking
    rate from a handful of customers is not mistaken for a pattern. Sorted from the highest
    rate, unless `sort` is False (for groups with a natural order, such as age bands).
    """
    table = df.groupby(column, observed=True)[target].agg(customers="size", subscribed="sum")
    intervals = [
        stats.binomtest(int(k), int(n)).proportion_ci(method="wilson")
        for k, n in zip(table["subscribed"], table["customers"], strict=True)
    ]
    table = table.assign(
        rate=table["subscribed"] / table["customers"],
        low=[ci.low for ci in intervals],
        high=[ci.high for ci in intervals],
    )
    return table.sort_values("rate", ascending=False) if sort else table