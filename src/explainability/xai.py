from lime.lime_text import LimeTextExplainer
import joblib

if __name__ == '__main__':

    folder = "nyt_data"  # [complete, reddit_data, bbc_data, nyt_data]

    model = joblib.load(f"../sample_code/sentiment_model_{folder}.pkl")

    target_names = ['negative', 'neutral', 'positive']

    inp = "Afghanistan and most of it neighbor agreed at a Paris conference to work together to stability the country " \
          "restrict narcotic traffic and coordinate action against terrorist group"
    explainer = LimeTextExplainer(class_names=["negative", "neutral", "positive"])
    exp = explainer.explain_instance(inp, model.predict_proba, num_features=6, labels=[0, 1, 2])

    print('Explanation for class %s' % target_names[0])
    print('\n'.join(map(str, exp.as_list(label=0))))
    print()
    print('Explanation for class %s' % target_names[1])
    print('\n'.join(map(str, exp.as_list(label=1))))
    print()
    print('Explanation for class %s' % target_names[2])
    print('\n'.join(map(str, exp.as_list(label=2))))

    exp.save_to_file(f'expl_{folder}.html')
