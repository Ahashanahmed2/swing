result_df.insert(0, "no", range(1, len(result_df) + 1))
        
        # Format dates for output
        result_df["date"] = pd.to_datetime(result_df["date"]).dt.strftime("%Y-%m-%d")
        result_df["p1_date"] = pd.to_datetime(result_df["p1_date"]).dt.strftime("%Y-%m-%d")
        result_df["p2_date"] = pd.to_datetime(result_df["p2_date"]).dt.strftime("%Y-%m-%d")
        
        # Final column order
        result_df = result_df[full_cols]
    else:
        result_df = pd.DataFrame(columns=full_cols)
else:
    result_df = pd.DataFrame(columns=full_cols)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
result_df.to_csv(output_path, index=False)

print(f"✅ ai_signal/buy.csv updated with {len(result_df)} signals:")
if len(result_df) > 0:
    print(f"   📈 Top RRR: {result_df['RRR'].max():.2f} | Avg RRR: {result_df['RRR'].mean():.2f}")
    print(f"   📉 Min diff: {result_df['diff'].min():.4f}")
    print(f"   💰 Avg position: {result_df['position_size'].mean():.0f} shares")
    print(f"   🎯 Avg actual risk: {result_df['actual_risk_bdt'].mean():,.0f} BDT")
    print(f"   📅 p1_date range: {result_df['p1_date'].min()} to {result_df['p1_date'].max()}")
    print(f"   📅 p2_date range: {result_df['p2_date'].min()} to {result_df['p2_date'].max()}")
else:
    print("   ⚠️ No valid signals found")