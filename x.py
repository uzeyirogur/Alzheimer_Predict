import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("data/alzheimer.csv")

null = data.isnull().sum()
data = data.fillna(method='ffill')  # Eksik değerleri önceki değerle doldurabilirsiniz

correlation_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
plt.title('Korelasyon Matrisi', fontsize=22)
plt.show()

plt.figure(figsize=(20,20))
sns.lineplot(x="Age",y="Group",color="blue",data=data)
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='SES', y='Group', data=data)
plt.title('SES Hastalık İlişkisi')
plt.xlabel('yaş')
plt.ylabel('grup')
plt.show()


pd.crosstab(data.Age,data.Group).plot(kind="bar",figsize=(20,6))
plt.title('Yaşa Göre Alzheimer Grafiği')
plt.xlabel('Age')
plt.ylabel('Group')
plt.show()

pd.crosstab(data['M/F'], data['Group']).plot(kind="bar", figsize=(15, 6), color=['#1CA53B', '#AA1111'])
plt.title('Yaşa Göre Alzheimer Grafiği')
plt.xlabel('Cinsiyet (0 = Kadın, 1 = Erkek)')
plt.legend(["Alzheimer Değil", "Alzheimer Hastası"])
plt.ylabel('Sayı')
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x="MMSE", y="Group", data=data, ci=None)
plt.xlabel("MMSE")
plt.ylabel("Group")
plt.title("MMSE ve grup un ilişkisi")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="CDR", y="Group", data=data, ci=None)
plt.xlabel("CDR")
plt.ylabel("Group")
plt.title("CDR ve grup un ilişkisi")
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x="nWBV", y="Group", data=data, ci=None,color="#FF0000")
plt.xlabel("nWBV")
plt.ylabel("Group")
plt.title("nWBV ve grup un ilişkisi")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='SES', y='Group', data=data)
plt.title('SES ve grup ilişkileri')
plt.xlabel('SES')
plt.ylabel('Group')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='EDUC', y='Group', data=data)
plt.title('eğitim ve grup İlişkisi')
plt.xlabel('Egitim')
plt.ylabel('Group')
plt.show()

# kategorik veriyi numeriğe çevirme
from sklearn.preprocessing import LabelEncoder
hastalik_durumu=data["Group"]
hastalik_df=pd.DataFrame(hastalik_durumu,columns=["Group"])

label_encoder=LabelEncoder()
hastalik_df["hastalik_kategorileri_sklearn"]=label_encoder.fit_transform(hastalik_df["Group"])

#numerik verileri 0 ile 1 aralığına dönüştür
import numpy as np
from sklearn.preprocessing import OneHotEncoder
data['M/F'] = label_encoder.fit_transform(data['M/F'])
data['Group'] = label_encoder.fit_transform(data['Group'])


hasta_durumu = ["hasta degil", "hasta", "ileri derecede hasta"]
hasta_df = pd.DataFrame(hasta_durumu, columns=["Group"])

enc = OneHotEncoder(handle_unknown="ignore")

enc_result = enc.fit_transform(hasta_df[["Group"]])  # "Group" sütununu seç
data['M/F'] = label_encoder.fit_transform(data['M/F'])
data['Group'] = label_encoder.fit_transform(data['Group'])
enc_df = pd.DataFrame(enc_result.toarray(), columns=enc.get_feature_names_out(["Group"]))

print(enc_df)

hastalik_df=hastalik_df.join(enc_df)

dummy_df=pd.get_dummies(hasta_df,columns=["Group"])


X = data.drop('Group', axis=1)
y = data['Group']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Oranı: {accuracy:.2f}")


import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R^2 Skoru: {r2:.2f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


print(data.isnull().sum())