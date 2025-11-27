# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------
latest_df['No'] = range(1, len(latest_df) + 1)
latest_df['price'] = latest_df['close']
latest_df['value_traded'] = latest_df['value']

final_df = latest_df[['No','date','symbol','price','Avolume','TR','liquidity_rating']]

# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------
os.makedirs("./csv", exist_ok=True)
os.makedirs("./output/ai_signal", exist_ok=True)

final_df.to_csv("./csv/liquidity.csv", index=False)
final_df.to_csv("./output/ai_signal/liquidity.csv", index=False)

print("✅ Ultra-Fast & Realistic liquidity.csv generated!")
print(final_df.head())