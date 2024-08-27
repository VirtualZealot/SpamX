from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import config as cfg

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

def train_lr_model(x_train, y_train):
    # Initialize the LogisticRegression classifier
    model = LogisticRegression(**cfg.lr_params)
    # Train the Model
    model.fit(x_train, y_train)
    return model

def train_rf_model(x_train, y_train):
    # Initialize the RandomForest classifier
    model = RandomForestClassifier(**cfg.rf_params)
    # Train the Model
    model.fit(x_train, y_train)
    return model

def train_xgb_model(x_train, y_train):
    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier(**cfg.xgb_params)
    # Train the model
    model.fit(x_train, y_train)
    return model