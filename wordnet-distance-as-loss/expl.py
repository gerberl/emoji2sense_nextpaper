df = pd.read_json('./emoji_synsets_results.json', lines=True, orient='records')
df.columns = ['y_true', 'y_pred']

# https://www.nltk.org/howto/wordnet.html
from nltk.corpus import wordnet as wn

df['sim'] = df.apply(lambda x: wn.synset(x['y_true']).path_similarity(wn.synset(x['y_pred'])), axis='columns')

path_sim_df = df.loc[ df['sim'].notnull() ]
path_sim_df.describe()
path_sim_df.sort_values()

df.sort_values('sim').head(10)
wn.synset('solicitation.n.03').definition()