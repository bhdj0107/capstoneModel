import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 파라미터 설정
trainRatio = 0.8

# 가격, 물량 정보 데이터 스케일링 및 가공
df = pd.read_csv('../data/선어_갈치가격_nondrop.csv', encoding='cp949')
df = df.drop('amount', axis=1)
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.weekday
df = df.iloc[:, 3:]

trainSize = int(len(df) * trainRatio)
trainX_temp = df.iloc[:trainSize, :]
testX_temp = df.iloc[trainSize:-1, :]

answer = df['middle_cost'].shift(-1).dropna()
trainY_temp = answer.iloc[:trainSize]
testY_temp = answer.iloc[trainSize:]

Xscaler = MinMaxScaler()
trainX = Xscaler.fit_transform(trainX_temp.iloc[:, :5])
trainX = np.hstack([trainX, trainX_temp['weekday'].to_numpy().reshape(-1, 1)])

testX = Xscaler.transform(testX_temp.iloc[:, :5])
testX = np.hstack([testX, testX_temp['weekday'].to_numpy().reshape(-1, 1)])

Yscaler = MinMaxScaler()
trainY = Yscaler.fit_transform(trainY_temp.to_numpy().reshape(-1, 1))
testY = Yscaler.transform(testY_temp.to_numpy().reshape(-1, 1))

# 날씨 정보 데이터 스케일링 및 가공
weather = pd.read_csv('../data/일별3해날씨평균정보.csv', encoding='cp949').iloc[3:, 1:-1]
weatherScaler = MinMaxScaler()

trainWeather = weather.iloc[:trainSize * 3, :].to_numpy()
testWeather = weather.iloc[trainSize * 3:-3, :].to_numpy()

weatherScaler = MinMaxScaler()
trainWeather = weatherScaler.fit_transform(trainWeather)
testWeather = weatherScaler.transform(testWeather)

trainWeather = trainWeather.reshape(-1, 3 * 9)
testWeather = testWeather.reshape(-1, 3 * 9)

# 날씨 + 가격 = 인풋 데이터로 가공
trainX = np.hstack([trainX, trainWeather])
testX = np.hstack([testX, testWeather])

# 모델 훈련
model = LinearRegression()
model.fit(trainX, trainY)

# 예측
predY = Yscaler.inverse_transform(model.predict(testX))
testY = Yscaler.inverse_transform(testY)

import warnings
warnings.filterwarnings(action='ignore')

# 그래프로 시각화 (예측 결과 포함)
plt.figure(figsize=(20, 10))
plotTrainData = Yscaler.inverse_transform(trainY).flatten()[-100:]
plt.plot(range(100), plotTrainData, label='Train Data[-100:]')

outputSize = predY.shape[0]
plt.plot(range(100, 100 +outputSize), testY.flatten(), label='Future Predictions', linestyle='--')
plt.plot(range(100, 100 + outputSize), predY.flatten(), label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices and Future Predictions')
plt.legend()
plt.show()

# 평가 지표 계산
mse = mean_squared_error(testY, predY)
rmse = np.sqrt(mse)
mae = mean_absolute_error(testY, predY)
r2 = r2_score(testY, predY)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

import joblib
joblib.dump(model, 'linear_path.pth')

joblib.dump({
    'xscaler': Xscaler,
    'yscaler': Yscaler,
    'weatherScaler': weatherScaler,
}, 'scalers.bin')