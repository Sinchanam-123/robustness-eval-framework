# Adult Census Income — EDA notes (Phase 1, Day 2)

## Shape
- rows: 32561, columns: 15

## Dtypes
- age: int64
- workclass: str
- fnlwgt: int64
- education: str
- education-num: int64
- marital-status: str
- occupation: str
- relationship: str
- race: str
- sex: str
- capital-gain: int64
- capital-loss: int64
- hours-per-week: int64
- native-country: str
- income: str

## Nulls (encoded as "?" in raw data)
- workclass: 1836 (5.64%)
- occupation: 1843 (5.66%)
- native-country: 583 (1.79%)

## Class balance (income)
- <=50K: 0.7592
- >50K: 0.2408

Note: real class imbalance (~76%/24%) — this is why accuracy alone is not sufficient and precision/recall/F1 are tracked too.

## Numeric feature ranges (min / max / mean / std)
These std values are what stress-test severity scaling multiplies against (e.g. `add_noise` uses `X.std(axis=0) * severity`) — recorded here so severity can be sanity-checked against real feature scale.

| feature | min | max | mean | std |
|---|---|---|---|---|
| age | 17.00 | 90.00 | 38.58 | 13.64 |
| fnlwgt | 12285.00 | 1484705.00 | 189778.37 | 105549.98 |
| education-num | 1.00 | 16.00 | 10.08 | 2.57 |
| capital-gain | 0.00 | 99999.00 | 1077.65 | 7385.29 |
| capital-loss | 0.00 | 4356.00 | 87.30 | 402.96 |
| hours-per-week | 1.00 | 99.00 | 40.44 | 12.35 |

## Categorical feature cardinality
- workclass: 8 unique values
- education: 16 unique values
- marital-status: 7 unique values
- occupation: 14 unique values
- relationship: 6 unique values
- race: 5 unique values
- sex: 2 unique values
- native-country: 41 unique values
