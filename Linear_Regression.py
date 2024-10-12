import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

Data_Path = os.path.join("AU Housing Data", "Cleaned - AU Real Estate Data.csv")
House_List = pd.read_csv(Data_Path)

def Model_Saving(Model, Scaler):
    Main_Path = os.getcwd()
    if os.path.exists('Model Results'):
        files = glob.glob(os.path.join('Model Results', '*'))
        for f in files:
            os.remove(f)
    else:
        os.makedirs("Model Results", exist_ok=True) 
    os.chdir(os.path.join(Main_Path, 'Model Results'))        
    joblib.dump(Model,'Price_Prediction_Model.pkl',)
    joblib.dump(Scaler, 'Scaler.pkl')
    os.chdir(Main_Path)

def Model_Traning(House_List):
    X = House_List[['SqFt', 'Bedrooms']] 
    Y = House_List['Price']

    # Chuẩn hóa dữ liệu
    Scaler = StandardScaler()
    X = Scaler.fit_transform(X)

    # Tạo tập train và tập test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Huấn luyện model 
    Model = LinearRegression()
    Model.fit(X_train, Y_train)

    #Lưu Model và Scaler    
    Model_Saving(Model, Scaler)



