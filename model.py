from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier


class Model:

    def fit_model(self, df):

        independent_values = df.drop(columns=["Survived"], axis=1)
        dependent_value = df[["Survived"]]
        kf = KFold(n_splits=5,random_state=42,shuffle=True)
        for train_index,val_index in kf.split(independent_values):
            X_train,X_val = independent_values.iloc[train_index],independent_values.iloc[val_index]
            y_train,y_val = dependent_value.iloc[train_index],dependent_value.iloc[val_index]
        gradient_booster_model = GradientBoostingClassifier(learning_rate=0.1)
        gradient_booster_model.fit(X_train,y_train)
        return gradient_booster_model

    def main(self, df):
        model = self.fit_model(df)
        model.predict()