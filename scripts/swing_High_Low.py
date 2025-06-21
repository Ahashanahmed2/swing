import pandas as pd
from collections import defaultdict
#swing_High_List
swing_high_candles = defaultdict(list)
swing_high_confirms = defaultdict(list)


#Swing_Low_List
swing_low_candles = defaultdict(list)
import glob
swing_Low_confirms=defaultdict(list)
from swing_point import identify_swing_points
import os

#path:./swing_low_csv
os.makedirs('./csv/swing/swing_low/low_candle/',exist_ok=True)
os.makedirs('./csv/swing/swing_low/low_confirm/',exist_ok=True)

#path:./swing_high_csv/
os.makedirs('./csv/swing/swing_high/high_candle/',exist_ok=True)
os.makedirs('./csv/swing/swing_high/high_confirm/',exist_ok=True)



mongodb_data = pd.read_csv('./csv/mongodb.csv')
symbol_group= mongodb_data.groupby('symbol')
for symbol,df in symbol_group:

  swing_lows, swing_highs= identify_swing_points(df)

#Swing_High 
  if len(swing_highs)>0:
    for index_rows_,index_rows__ in swing_highs:
        swing_highs_candle_data=df.loc[index_rows_]
        swing_highs_confirm_data=df.loc[index_rows__]

       

            # প্রতিটি symbol এর জন্য লিস্টে অ্যাড করা
        swing_high_candles[symbol].append(swing_highs_candle_data)
        swing_high_confirms[symbol].append(swing_highs_confirm_data)
        

    for symbol in swing_high_candles:
      pd.DataFrame(swing_high_candles[symbol]).to_csv(f'./csv/swing/swing_high/high_candle/{symbol}.csv', index=False)
      pd.DataFrame(swing_high_confirms[symbol]).to_csv(f'./csv/swing/swing_high/high_confirm/{symbol}.csv', index=False)

#Swing_Low
  if len(swing_lows)>0:
    for index_row_,index_rows__ in swing_lows:
        swing_lows_candle_data=df.loc[index_row_]
        swing_lows_confirms_data=df.loc[index_rows__]

        # প্রতিটি symbol এর জন্য লিস্টে অ্যাড করা
        swing_low_candles[symbol].append(swing_lows_candle_data)
        swing_Low_confirms[symbol].append(swing_lows_confirms_data)


    for symbol in swing_low_candles:
        pd.DataFrame(swing_low_candles[symbol]).to_csv(f'./csv/swing/swing_low/low_candle/{symbol}.csv',index=False)
        pd.DataFrame(swing_Low_confirms[symbol]).to_csv(f'./csv/swing/swing_low/low_confirm/{symbol}.csv',index=False)
 
 
  






    