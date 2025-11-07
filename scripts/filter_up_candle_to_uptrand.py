import pandas as pd
from pathlib import Path

# ---------- 1. পথ ----------
up_path   = Path('./csv/up_candle.csv')
mongo_path= Path('./csv/mongodb.csv')

# ---------- 2. পড়া ----------
up_df    = pd.read_csv(up_path)
mongo_df = pd.read_csv(mongo_path)

# MongoDB-র প্রয়োজনীয় কলাম ডেট-টাইপ + ফ্লোট করা
mongo_df['date'] = pd.to_datetime(mongo_df['date'])
mongo_df['low']  = mongo_df['low'].astype(float)

# ---------- 3. symbol অনুযায়ী গ্রুপড ডিকশনারি ----------
# key = symbol, value = ওই symbol-এর সব রেকর্ড (ইতিমধ্যেই date সর্ট করা)
grouped_mongo = {
    sym: df.sort_values('date').reset_index(drop=True)
    for sym, df in mongo_df.groupby('symbol')
}

# ---------- 4. ফিল্টার ফাংশন ----------
def keep_row(up_row):
    symbol      = up_row['symbol']
    up_date     = pd.to_datetime(up_row['date'])
    up_low      = float(up_row['low'])
    ob_date     = pd.to_datetime(up_row['ob_low_last_date'])
    ob_low      = float(up_row['ob_low_last'])

    # MongoDB-তে symbol না থাকলে ফেলে দাও
    if symbol not in grouped_mongo:
        return False

    m_df = grouped_mongo[symbol]

    # শর্ত:
    # 1) up_date   < mongo_date
    # 2) up_low    > mongo_low
    # 3) ob_date   < mongo_date
    # 4) ob_low    < mongo_low
    mask = (
        (m_df['date'] > up_date)   &
        (m_df['low']  < up_low)    &
        (m_df['date'] > ob_date)   &
        (m_df['low']  > ob_low)
    )

    return mask.any()          # অন্তত একটি রেকর্ড মিললে True

# ---------- 5. ফিল্টার চালানো ----------
mask = up_df.apply(keep_row, axis=1)
filtered = up_df[mask].copy()

# ---------- 6. সেভ ----------
filtered.to_csv(up_path, index=False)
print(f'Filtered up_candle.csv saved. Rows before: {len(up_df)}, after: {len(filtered)}')
