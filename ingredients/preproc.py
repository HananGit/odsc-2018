from sacred import Ingredient


preprocess_ingredient = Ingredient('preprocess')


@preprocess_ingredient.config
def cfg():
    features = ['Fare', 'SibSp', 'Parch']


@preprocess_ingredient.named_config
def variant_simple():
    features = ['Fare', 'SibSp']


@preprocess_ingredient.named_config
def variant_all():
    features = '*'


@preprocess_ingredient.capture
def preprocess_data(df, features):
    if features == '*':
        return df.fillna(0.)
    else:
        return df[features].fillna(0.)
