# imports
from TaxiFareModel.data import get_data, clean_data, holdout_data
from TaxiFareModel.encoders import DistanceTransformer, ManhattenDistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        
        self.use_manhatten=False
        self.estimator="LinearRegression"
        
        self.mlflow_uri = "https://mlflow.lewagon.co/"
        self.experiment_name = "[DE] [Berlin] [domzae] TaxiFare + 1"

    def build_preproc_pipe(self):
        """builds the preprocessing pipeline"""
        dist = ManhattenDistanceTransformer() if self.use_manhatten else DistanceTransformer()
    
        dist_pipe = Pipeline([
            ('dist_trans', dist),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        return preproc_pipe

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        estimator_dict = {
            "LinearRegression": LinearRegression(),
            "SGDRegressor": SGDRegressor(),
            "Lasso": Lasso(),
            "Ridge": Ridge()
        }
        
        # default to LinearRegression
        if self.estimator not in estimator_dict.keys():
            self.estimator = "LinearRegression"

        preproc_pipe = self.build_preproc_pipe()
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            (self.estimator, estimator_dict[self.estimator])
        ])

        return self.pipeline

    def run(self, **kwargs):
        """set and train the pipeline"""
        
        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
        
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)

        return rmse
    
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X_train, X_test, y_train, y_test = holdout_data(df)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)
    #trainer.mlflow_log_param("model", "linear regression")
    #trainer.mlflow_log_metric("rmse", rmse)
    print(f'rmse: {rmse}')

    trainer.save_model()