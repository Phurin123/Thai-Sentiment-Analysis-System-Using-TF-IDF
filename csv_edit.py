# add_difficulty.py
import pandas as pd
import re

# Load data
df = pd.read_csv("data/1.synthetic_wisesight_like_thai_sentiment_5000.csv")

# Define rules for "hard" examples
def classify_difficulty(text):
    text = str(text)
    
    # Hard cases: mixed signal, negation + positive word, etc.
    has_neg = any(w in text for w in ["ไม่", "แย่กว่าที่คิด", "ผิดหวัง", "ควรปรับปรุง"])
    has_pos = any(w in text for w in ["ดีมาก", "ประทับใจ", "โอเคเกินคาด", "ชอบ"])
    has_ambiguous = any(phrase in text for phrase in ["ไม่ได้แย่ แต่ก็ไม่ได้ดีมาก", "อยู่ในระดับปกติ", "เฉยๆ"])
    has_slang_or_emoji = bool(re.search(r'[😤🙄😒🙂😊👍👎🤔😐]', text)) or "แม่ง" in text
    
    if (has_neg and has_pos) or has_ambiguous:
        return "hard"
    elif has_slang_or_emoji:
        return "noisy"
    else:
        return "easy"

# Apply
df["difficulty"] = df["text"].apply(classify_difficulty)

# Save back (overwrite หรือ save เป็นไฟล์ใหม่)
df.to_csv("data/1.synthetic_wisesight_like_thai_sentiment_5000_with_difficulty.csv", index=False)

print("✅ เพิ่มคอลัมน์ 'difficulty' เรียบร้อย!")
print(df["difficulty"].value_counts())