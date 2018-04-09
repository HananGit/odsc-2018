from sacred import Ingredient


preprocess_ingredient = Ingredient('preprocess')


@preprocess_ingredient.config
def cfg():
    features = ['Fare', 'SibSp', 'Parch']


@preprocess_ingredient.named_config
def variant_preprocess_data():
    features = ['Fare', 'SibSp']


@preprocess_ingredient.capture
def preprocess_data(df, features):
    return df[features]
