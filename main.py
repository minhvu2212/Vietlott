import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from io import StringIO

# Hàm chuyển đổi ngày tháng tùy chỉnh
def custom_date_parser(date_str):
    date_str = date_str.split(', ')[1]
    return pd.to_datetime(date_str, format='%d/%m/%Y')

# Đọc dữ liệu
file_path = 'paste.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

df = pd.read_csv(StringIO(data), sep='\t')

# Tìm cột chứa kết quả xổ số
result_column = df.columns[df.apply(lambda x: x.dtype == 'object' and x.str.contains(r'\d+ \d+ \d+ \d+ \d+ \d+ \d+').any())][0]

# Tách dữ liệu kết quả
df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'NumExtra']] = df[result_column].str.split(expand=True)
number_columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'NumExtra']
df[number_columns] = df[number_columns].astype(int)

# Xử lý ngày tháng
df['Date'] = df['Ngày Mở Thưởng'].apply(custom_date_parser)
df.set_index('Date', inplace=True)

# 1. Phân tích xu hướng theo thời gian
plt.figure(figsize=(15, 8))
for col in number_columns:
    df[col].resample('ME').mean().plot(label=col)
plt.title('Xu hướng trung bình của các số theo thời gian')
plt.legend()
plt.show()

# 2. Phân tích cặp số
pair_counts = Counter((row.iloc[i], row.iloc[j]) 
                      for _, row in df[number_columns].iterrows() 
                      for i in range(len(number_columns)) 
                      for j in range(i+1, len(number_columns)))

top_pairs = pair_counts.most_common(10)
print("Top 10 cặp số xuất hiện nhiều nhất:")
for pair, count in top_pairs:
    print(f"Cặp số {pair}: {count} lần")

# 3. Kiểm định tính ngẫu nhiên
observed_freq = df[number_columns].values.flatten()
observed_counts = pd.Series(observed_freq).value_counts().sort_index()
expected_counts = pd.Series(np.ones(55) * len(observed_freq) / 55, index=range(1, 56))

chi2_result = chi2_contingency([observed_counts, expected_counts])
print(f"Chi-square test p-value: {chi2_result.pvalue}")

# 4. Phân tích khoảng cách giữa các lần xuất hiện
def calculate_gaps(series):
    return series.index[series].to_series().diff().dt.days.dropna()

gap_data = {num: calculate_gaps((df[number_columns] == num).any(axis=1)) for num in range(1, 56)}

plt.figure(figsize=(15, 8))
for num in range(1, 56):
    if not gap_data[num].empty:
        sns.kdeplot(gap_data[num], label=f'Số {num}')
plt.title('Phân phối khoảng cách giữa các lần xuất hiện')
plt.xlabel('Số ngày')
plt.ylabel('Mật độ')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5. Mô hình hóa xác suất (ARIMA)
def fit_arima(series):
    model = ARIMA(series, order=(1,1,1))
    results = model.fit()
    return results.forecast(steps=10)

arima_forecasts = {col: fit_arima(df[col]) for col in number_columns}
print("Dự đoán ARIMA cho 10 kỳ tiếp theo:")
for col, forecast in arima_forecasts.items():
    print(f"{col}: {forecast.tolist()}")

# 6. Phân tích theo vị trí
position_probs = df[number_columns].apply(lambda x: x.value_counts(normalize=True))

plt.figure(figsize=(12, 8))
sns.heatmap(position_probs, annot=True, cmap='YlGnBu')
plt.title('Xác suất xuất hiện của các số theo vị trí')
plt.show()

# 7. Tối ưu hóa hiệu suất (sử dụng numpy)
all_numbers = df[number_columns].values.flatten()
unique, counts = np.unique(all_numbers, return_counts=True)
probabilities = dict(zip(unique, counts / len(all_numbers)))

# 8. Phân cụm số (K-means)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df[number_columns])

df['Cluster'] = kmeans.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Num1', y='Num2', hue='Cluster', palette='deep')
plt.title('Phân cụm kết quả xổ số')
plt.show()

# 9. Tích hợp dữ liệu bổ sung (giả sử có cột 'Giải Jackpot 1')
if 'Giải Jackpot 1' in df.columns:
    plt.figure(figsize=(12, 6))
    df['Giải Jackpot 1'].plot()
    plt.title('Xu hướng giá trị Jackpot 1 theo thời gian')
    plt.ylabel('Giá trị Jackpot 1')
    plt.show()

    correlation = df[number_columns + ['Giải Jackpot 1']].corr()['Giải Jackpot 1'].sort_values(ascending=False)
    print("Tương quan giữa các số và giá trị Jackpot 1:")
    print(correlation)

# 10. Áp dụng học máy (Random Forest cho dự đoán)
X = df[number_columns]
y = df['Giải Jackpot 1'] if 'Giải Jackpot 1' in df.columns else df['Num1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Độ chính xác của mô hình Random Forest:", rf_model.score(X_test, y_test))

# Hiển thị tầm quan trọng của các đặc trưng
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Tầm quan trọng của các đặc trưng:")
print(feature_importance)

# 11. Dự đoán xác suất xuất hiện trong 10 đợt tiếp theo
future_draws = 10
future_probabilities = {num: 1 - (1 - prob) ** (7 * future_draws) for num, prob in probabilities.items()}

sorted_probabilities = sorted(future_probabilities.items(), key=lambda x: x[1], reverse=True)

print("\nXác suất xuất hiện của các số trong 10 đợt quay tiếp theo:")
for num, prob in sorted_probabilities:
    print(f"Số {num}: {prob*100:.2f}%")

# 12. Biểu đồ xác suất xuất hiện trong tương lai
plt.figure(figsize=(12, 6))
plt.bar([str(num) for num, _ in sorted_probabilities], [prob for _, prob in sorted_probabilities])
plt.title("Xác suất xuất hiện của các số trong 10 đợt quay tiếp theo")
plt.xlabel("Số")
plt.ylabel("Xác suất")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 13. Thống kê cơ bản
print("\nThống kê cơ bản:")
print(df[number_columns].describe())

# 14. Biểu đồ tần suất xuất hiện của các số
plt.figure(figsize=(12, 6))
all_numbers = df[number_columns].values.flatten()
plt.hist(all_numbers, bins=55, range=(1, 56), edgecolor='black')
plt.title('Tần suất xuất hiện của các số')
plt.xlabel('Số')
plt.ylabel('Số lần xuất hiện')
plt.show()

# 15. Phân tích xu hướng theo thời gian của từng số
for num in range(1, 56):
    df[f'Num_{num}'] = (df[number_columns] == num).any(axis=1).astype(int)

trends = df[[f'Num_{i}' for i in range(1, 56)]].resample('ME').mean()

plt.figure(figsize=(15, 8))
for col in trends.columns:
    plt.plot(trends.index, trends[col], label=col)
plt.title('Xu hướng xuất hiện của các số theo thời gian')
plt.xlabel('Thời gian')
plt.ylabel('Tần suất xuất hiện')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()