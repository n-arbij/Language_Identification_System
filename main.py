import pandas as pd

def load_text_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

swahili_texts = load_text_file("swahili.txt")
sheng_texts   = load_text_file("sheng.txt")
luo_texts     = load_text_file("luo.txt")
english_texts = load_text_file("english.txt")

df_swahili = pd.DataFrame({"text": swahili_texts, "language": "Swahili"})
df_sheng   = pd.DataFrame({"text": sheng_texts,   "language": "Sheng"})
df_luo     = pd.DataFrame({"text": luo_texts,     "language": "Luo"})
df_english = pd.DataFrame({"text": english_texts, "language": "English"})

df = pd.concat([df_swahili, df_sheng, df_luo, df_english], ignore_index=True)

df["text"] = df["text"].str.strip()
df = df[df["text"] != ""]
df = df.dropna(subset=["text"])
df = df.drop_duplicates(subset="text")
df = df[df["text"].str.split().str.len() >= 3]
df = df.reset_index(drop=True)
print(f"\nTotal samples after cleaning: {len(df)}")
print(df["language"].value_counts())

df.to_csv("language_dataset.csv", index=False)
print("\n✅ Dataset saved as language_dataset.csv")