import json
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import opencc

print("正在加载数据...")
with open('data/douban.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = []
labels = []
for item in data:
    rating = item.get('label')
    comment = item.get('text', '')
    
    if rating == 5: 
        texts.append(comment)
        labels.append(1) # 1代表正面
    elif rating == 1: 
        texts.append(comment)
        labels.append(0) # 0代表负面

print(f"数据准备完毕，总数据量: {len(texts)} 条")

cc = opencc.OpenCC('t2s')
def preprocess(text):
    text = cc.convert(str(text)) # 确保是字符串
    return " ".join(jieba.cut(text))

print("正在进行分词处理（可能需要一两分钟，请耐心等待）...")
X_processed = [preprocess(t) for t in texts]
y = labels

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print("正在进行 TF-IDF 特征提取...")
vectorizer = TfidfVectorizer(max_features=2000) # 选择最常见的2000个词
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("开始训练模型...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("\n=== 测试集评估 ===")
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\n混淆矩阵 (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\n详细报告 (Precision, Recall, F1):")
print(classification_report(y_test, y_pred))

errors = []
for text, true, pred in zip(X_test, y_test, y_pred):
    if true != pred:
        errors.append({"text": text, "true_label": true, "predicted_label": pred})
pd.DataFrame(errors).to_csv("output/error_analysis.csv", index=False, encoding='utf-8-sig')
print("\n成功！错误样本已保存到 output/error_analysis.csv")

print("\n=== 开始分析新文本情感（骆驼祥子） ===")
novel_file = 'data/luotuoxiangzi.txt' 
try:
    with open(novel_file, 'r', encoding='utf-8') as f:
        novel_text = f.read()

    def split_text(text, chunk_size=500):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    chunks = split_text(novel_text)
    print(f"小说共切分为 {len(chunks)} 个片段。")

    novel_probs = []
    for chunk in chunks:
        chunk_vec = vectorizer.transform([preprocess(chunk)])
        prob = model.predict_proba(chunk_vec)[0][1] 
        novel_probs.append(prob)

    pd.DataFrame({"chunk_index": range(len(novel_probs)), "sentiment_prob": novel_probs}).to_csv("output/novel_sentiment.csv", index=False)
    print("成功！小说情感概率已保存到 output/novel_sentiment.csv")
except Exception as e:
    print(f"分析小说时出错: {e}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(novel_probs)
plt.title("Emotional Arc of Luo Tuo Xiangzi")
plt.xlabel("Chunk Index")
plt.ylabel("Positive Probability")
plt.savefig("output/sentiment_curve.png")
print("图片已保存到 output/sentiment_curve.png")